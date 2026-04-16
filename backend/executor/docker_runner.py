"""
Docker Runner – Executes fix scripts inside isolated Docker containers.
"""
import asyncio
import logging
import time
import os
import re
import docker
from docker.errors import DockerException

from backend.config import settings

logger = logging.getLogger(__name__)


FIX_CONTAINER_IMAGE = "pipegenie-runner:latest"


class DockerRunner:
    def __init__(self):
        try:
            self.client = docker.from_env()
            self._ensure_image()
        except DockerException as e:
            logger.warning(f"[DockerRunner] Docker not available: {e}")
            self.client = None

    def _ensure_image(self):
        """Build the fix runner image if not present."""
        try:
            self.client.images.get(FIX_CONTAINER_IMAGE)
            logger.info("[DockerRunner] Runner image found")
        except Exception:
            logger.info("[DockerRunner] Building runner image...")
            dockerfile_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "docker", "fix_runner"
            )
            if os.path.exists(dockerfile_path):
                self.client.images.build(
                    path=os.path.abspath(dockerfile_path),
                    tag=FIX_CONTAINER_IMAGE,
                    rm=True
                )
            else:
                logger.warning("[DockerRunner] Dockerfile not found, using base image")

    async def run_fix(self, fix_script: str, repo_url: str,
                      branch: str, event_id: str, repo_full_name: str) -> dict:
        """
        Execute a fix script inside a Docker container.
        Returns: {"exit_code": int, "output": str, "duration": float, "container_id": str}
        """
        if not self.client:
            return await self._simulate_execution(fix_script)

        start_time = time.time()
        container_id = None
        fix_branch = self._build_fix_branch(event_id)

        try:
            if settings.REPO_WRITEBACK_ENABLED:
                # Execute generated fix script against a cloned repository checkout.
                script_content = f"""#!/bin/bash
set -e
echo "=== PipeGenie Fix Runner ==="
echo "Event: {event_id}"
echo "Repo: {repo_url}"
echo "Branch: {branch}"
echo "Fix Branch: {fix_branch}"
echo "================================"

if [ -z "$REPO_FULL_NAME" ]; then
    echo "Missing REPO_FULL_NAME"
    exit 1
fi

AUTH_REPO_URL="https://github.com/$REPO_FULL_NAME.git"
if [ -n "$GITHUB_TOKEN" ]; then
    AUTH_REPO_URL="https://x-access-token:$GITHUB_TOKEN@github.com/$REPO_FULL_NAME.git"
fi

rm -rf /workspace/repo
git clone --depth 1 --branch "$BRANCH" "$AUTH_REPO_URL" /workspace/repo
cd /workspace/repo

git config user.name "$PIPEGENIE_BOT_NAME"
git config user.email "$PIPEGENIE_BOT_EMAIL"
git checkout -b "$FIX_BRANCH"

{fix_script}

if git diff --quiet && git diff --cached --quiet; then
    echo "No repository changes detected after fix script"
    echo "PIPEGENIE_FIX_BRANCH=$FIX_BRANCH"
    exit 0
fi

git add -A
git commit -m "chore(pipegenie): apply generated fix for event {event_id}"
git push origin "$FIX_BRANCH"

echo "PIPEGENIE_FIX_BRANCH=$FIX_BRANCH"

echo "=== Fix Script Completed ==="
"""
            else:
                script_content = f"""#!/bin/bash
set -e
echo "=== PipeGenie Script-Only Runner ==="
echo "Event: {event_id}"
echo "Repo: {repo_url}"
echo "Branch: {branch}"
echo "================================"

{fix_script}

echo "=== Fix Script Completed ==="
"""

            script_content = self._sanitize_script(script_content)

            use_runner_image = self._image_exists()
            image_name = FIX_CONTAINER_IMAGE if use_runner_image else "python:3.11-slim"
            # pipegenie-runner image sets ENTRYPOINT to /bin/bash, so pass only args.
            command = ["-c", script_content] if use_runner_image else ["bash", "-c", script_content]

            # Run container
            container = self.client.containers.run(
                image=image_name,
                command=command,
                environment={
                    "REPO_URL": repo_url,
                    "REPO_FULL_NAME": repo_full_name,
                    "BRANCH": branch,
                    "FIX_BRANCH": fix_branch,
                    "EVENT_ID": event_id,
                    "PYTHONUNBUFFERED": "1",
                    "GITHUB_TOKEN": settings.GITHUB_TOKEN,
                    "PIPEGENIE_BOT_NAME": settings.PIPEGENIE_BOT_NAME,
                    "PIPEGENIE_BOT_EMAIL": settings.PIPEGENIE_BOT_EMAIL,
                },
                mem_limit="512m",
                cpu_period=100000,
                cpu_quota=50000,  # 50% CPU
                network_mode="bridge",
                remove=True,
                detach=False,
                stdout=True,
                stderr=True
            )

            duration = time.time() - start_time
            output = container.decode("utf-8") if isinstance(container, bytes) else str(container)
            parsed_fix_branch = self._extract_fix_branch(output) or (fix_branch if settings.REPO_WRITEBACK_ENABLED else None)

            return {
                "exit_code": 0,
                "output": output,
                "duration": duration,
                "container_id": container_id or "completed",
                "fix_branch": parsed_fix_branch,
            }

        except docker.errors.ContainerError as e:
            duration = time.time() - start_time
            stderr = e.stderr.decode("utf-8") if e.stderr else str(e)
            parsed_fix_branch = self._extract_fix_branch(stderr) or (fix_branch if settings.REPO_WRITEBACK_ENABLED else None)
            return {
                "exit_code": e.exit_status,
                "output": stderr,
                "duration": duration,
                "container_id": container_id or "failed",
                "fix_branch": parsed_fix_branch,
            }
        except Exception as e:
            logger.error(f"[DockerRunner] Unexpected error: {e}")
            return {
                "exit_code": 1,
                "output": f"Docker execution failed: {str(e)}",
                "duration": time.time() - start_time,
                "container_id": "error",
                "fix_branch": fix_branch if settings.REPO_WRITEBACK_ENABLED else None,
            }

    def _build_fix_branch(self, event_id: str) -> str:
        safe = re.sub(r"[^a-zA-Z0-9-]", "-", event_id)[:24]
        return f"pipegenie/fix-{safe}"

    def _extract_fix_branch(self, output: str) -> str | None:
        match = re.search(r"PIPEGENIE_FIX_BRANCH=([^\s]+)", output or "")
        return match.group(1) if match else None

    def _sanitize_script(self, script: str) -> str:
        """Fix known shell-command incompatibilities in model-generated scripts."""
        sanitized = script
        replacements = {
            "--max-redirects": "--max-redirs",
        }

        for old, new in replacements.items():
            if old in sanitized:
                sanitized = sanitized.replace(old, new)
                logger.warning(
                    f"[DockerRunner] Replaced unsupported flag '{old}' with '{new}'"
                )

        return sanitized

    def _image_exists(self) -> bool:
        try:
            self.client.images.get(FIX_CONTAINER_IMAGE)
            return True
        except Exception:
            return False

    async def _simulate_execution(self, fix_script: str) -> dict:
        """Simulate execution when Docker is not available (dev mode)."""
        logger.warning("[DockerRunner] Simulating Docker execution (Docker not available)")
        await asyncio.sleep(2)  # Simulate execution time

        # Basic script validation
        dangerous = ["rm -rf /", "DROP TABLE", "DELETE FROM"]
        for d in dangerous:
            if d in fix_script:
                return {
                    "exit_code": 1,
                    "output": f"BLOCKED: Dangerous command detected: {d}",
                    "duration": 2.0,
                    "container_id": "simulated-blocked"
                }

        return {
            "exit_code": 0,
            "output": f"[SIMULATED] Fix script executed successfully:\n{fix_script[:300]}\n[DONE]",
            "duration": 2.0,
            "container_id": "simulated-success",
            "fix_branch": "pipegenie/fix-simulated"
        }

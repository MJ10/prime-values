# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "cluster-uv @ git+https://github.com/mila-iqia/cluv.git@2b1a69e",
# ]
# ///

from __future__ import annotations

import logging
import shlex
import subprocess
from importlib import import_module
from pathlib import PurePosixPath

from cluv.cache import ProjectStateOnCluster
from cluv.remote import Remote

logger = logging.getLogger("cluv")
cluv_sync = import_module("cluv.cli.sync")


async def sync_prime_rl(
    remote: Remote,
    project_path: PurePosixPath,
    project_state: ProjectStateOnCluster,
) -> None:
    current_git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if project_state.last_uv_sync_git_commit == current_git_commit:
        logger.info(
            "Submodules and all extras are already synced for commit %s on %s.",
            current_git_commit,
            remote.hostname,
        )
        return

    quoted_project_path = shlex.quote(str(project_path))
    await remote.run(
        f"git -C {quoted_project_path} submodule sync --recursive",
        hide=True,
    )
    await remote.run(
        f"git -C {quoted_project_path} submodule update --init --recursive",
    )
    uv_command = f"uv --directory={quoted_project_path} sync --all-extras"
    await remote.run(f"bash --login -c {shlex.quote(uv_command)}")
    project_state.last_uv_sync_git_commit = current_git_commit


cluv_sync.run_uv_sync = sync_prime_rl

from cluv.__main__ import main  # noqa: E402

if __name__ == "__main__":
    main()

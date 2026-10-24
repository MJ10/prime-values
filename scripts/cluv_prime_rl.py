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
cluv_submit = import_module("cluv.cli.submit")
cluv_slurm = import_module("cluv.slurm")
# These clusters run glibc 2.34; the mooncake-transfer-engine wheel needs 2.35.
SYNC_EXCLUDED_PACKAGES = {
    cluster: ("mooncake-transfer-engine",) for cluster in ("fir", "nibi", "rorqual", "trillium-gpu")
}
# Run directories this large stay on the cluster when cluv fetches results.
RESULT_SYNC_EXCLUDES = ("checkpoints/", "broadcasts/", "weights/")
upstream_run = cluv_sync.run
upstream_run_saccts = cluv_slurm.run_saccts


async def run_excluding_large_results(command, *args, **kwargs):
    if command[0] == "rsync":
        command = (command[0], *(f"--exclude={pattern}" for pattern in RESULT_SYNC_EXCLUDES), *command[1:])
    return await upstream_run(command, *args, **kwargs)


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
    sync_args = ["sync", "--all-extras"]
    for package in SYNC_EXCLUDED_PACKAGES.get(remote.hostname, ()):
        sync_args.extend(("--no-install-package", package))
    uv_command = (
        f'uv --cache-dir "$SCRATCH/.cache/prime-values-uv" --directory={quoted_project_path} {shlex.join(sync_args)}'
    )
    await remote.run(f"bash --login -c {shlex.quote(uv_command)}")
    project_state.last_uv_sync_git_commit = current_git_commit


async def run_saccts_in_job_order(remote: Remote | None, jobs: list[int | None], format: str = "State") -> list[str]:
    """Upstream ``run_saccts`` returns sacct's lines in job-id order, but ``submit`` zips them
    with its submissions in submission order. With several submissions on one cluster (one per
    account), the states swap whenever a later submission gets the lower job id, and ``submit``
    then cancels the job that is actually running. Return one state per requested job, in the
    requested order; a job not yet visible to sacct reads as PENDING."""
    jobs = [job for job in jobs if job is not None]
    lines = await upstream_run_saccts(remote, jobs, format=f"JobID,{format}")
    states = dict(line.split("|", 1) for line in lines)
    return [states.get(str(job), "PENDING") for job in jobs]


cluv_sync.run_uv_sync = sync_prime_rl
cluv_submit.run_saccts = run_saccts_in_job_order
cluv_sync.run = run_excluding_large_results

from cluv.__main__ import main  # noqa: E402

if __name__ == "__main__":
    main()

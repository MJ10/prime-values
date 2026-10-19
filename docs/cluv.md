# Running on clusters with cluv

[cluv](https://github.com/mila-iqia/cluv) synchronizes uv projects and submits
Slurm jobs over SSH. This repository includes cluv configuration, a Slurm job
script, and a pinned launcher that prepares prime-rl's submodules and optional
dependencies on every remote checkout.

## Cluster defaults

Cluv uses the host aliases from `~/.ssh/config`. The repository configures the
currently supported GPU allocations as follows:

| Cluster | Account | Partition | GPU request |
| --- | --- | --- | --- |
| Mila | `mila` | `main` | 2× A100L |
| Tamia | `aip-bengioy` | `gpubase_bynode_b2` | one 4× H100 node |
| Fir | `rrg-bengioy-ad_gpu` or `def-bengioy_gpu` | `gpubase_bygpu_b2` | 2× H100 |

Mila's `main` QOS caps each user at 8 CPUs, 2 GPUs, and 48 GB, so the Mila
defaults request exactly that. For more, override `--partition`:
`short-unkillable` allows 4 GPUs with no CPU cap for up to 3 hours at high
priority, and `long` has no per-user cap but is preemptible. Mila GPU nodes
hold 4× A100L, 4× L40S, or 8× H100.

On Fir, cluv submits against both allocations and keeps whichever starts first.
The Tamia and Fir `b2` partitions match the four-hour default and accept jobs up
to 12 hours. Pair a longer `--time` with the appropriate partition.

Use cluv's remote shell to discover the settings for another connected cluster.
The output is prefixed with each cluster name:

```bash
uv run scripts/cluv_prime_rl.py sh sacctmgr -nP show assoc where \
  'user=$USER' format=Cluster,Account,Partition,QOS
uv run scripts/cluv_prime_rl.py sh sinfo -h -o %P,%l,%G
```

To add another SSH alias, define its allocation, partition, GPU request, and any
cluster-specific paths in `pyproject.toml`:

```toml
[tool.cluv.clusters.nibi]
results_path = "$SCRATCH/prime-values"
sbatch_args = { account = "def-mygroup_gpu", partition = "gpubase_bygpu_b2", gpus-per-node = "h100:2" }
```

Do not put tokens in `pyproject.toml`. Configure W&B, Hugging Face, and GitHub
credentials on each cluster. The remote login nodes also need access to this
repository and its git submodules.

The launcher is a PEP 723 script, so cluv stays isolated from prime-rl's Python
3.12 environment. Use it in place of a globally installed `cluv` command:

```bash
uv run scripts/cluv_prime_rl.py login mila
uv run scripts/cluv_prime_rl.py sync mila
```

`sync` checks out the current pushed commit, initializes all submodules, and
runs `uv sync --all-extras` on the cluster. Dependencies use the isolated
`$SCRATCH/.cache/prime-values-uv` cache instead of the shared home uv cache.
Cluv requires tracked changes to be committed before submission.

On Mila and Tamia, the remote checkout lives at `$SCRATCH/Projects/prime-values`
so that its `.venv` shares a filesystem with the uv cache and stays out of the
home quota. Other clusters default to the same path under `$HOME`; set
`project_dir` in a cluster's `[tool.cluv.clusters.<name>]` table to move it.

Tamia compute nodes reach the internet only through the cluster's Squid proxy.
Its `env` table exports `http_proxy`, `https_proxy`, and a `no_proxy` that keeps
localhost traffic between prime-rl components off the proxy; Hugging Face, PyPI,
W&B, and GitHub are reachable through it. A checkout with its `.venv` and uv
cache takes roughly 180K inodes, so check the scratch file quota with
`diskusage_report` before the first sync.

Fir's glibc 2.34 is older than the available `mooncake-transfer-engine` wheel
requires, so the Fir sync omits that package. Standard training and inference
runs work there; Mooncake-backed disaggregated inference does not.

## Submit a single-node experiment

Pass Slurm options before `--` and the prime-rl command after it. The defaults
request 16 CPUs, 128 GB of RAM, and four hours. GPU resources and the Mila CPU
and memory caps are cluster-specific; command-line values override them.

```bash
uv run scripts/cluv_prime_rl.py submit mila \
  --partition=long --gpus-per-node=a100l:4 --cpus-per-task=32 --mem=256G --time=12:00:00 \
  -- rl @ examples/hendrycks_sanity/rl.toml \
  --wandb.project my-project \
  --wandb.name hendrycks-cluv
```

Do not add a prime-rl Slurm overlay to this command: cluv already submits the
single-node allocation. The job script runs `uv run --no-sync` inside the
allocation and, unless `--output-dir` is supplied explicitly, places the full
run beside the Slurm log in cluv's results directory. Put any explicit output
directory under `cluv-results/` if it should be fetched by `cluv sync`.

SFT uses the same job script:

```bash
uv run scripts/cluv_prime_rl.py submit mila \
  --gpus-per-node=a100l:1 --time=2:00:00 \
  -- sft @ examples/reverse_text/sft.toml
```

Use `first` instead of a cluster name to submit to every connected,
non-disabled cluster and keep the first allocation that starts:

```bash
uv run scripts/cluv_prime_rl.py submit first \
  --time=4:00:00 \
  -- rl @ examples/reverse_text/rl.toml
```

## Submit a multi-node experiment

Prime-rl owns the multi-node Slurm topology. Use `cluv run` to synchronize the
checkout and invoke prime-rl's launcher on the remote login node. Start from a
config containing `[slurm]` and a multi-node `[deployment]` block:

```bash
uv run scripts/cluv_prime_rl.py run mila -- \
  --no-sync rl @ examples/multinode/rl.toml \
  --output-dir cluv-results/my-multinode-run
```

The leading `--no-sync` is an option to the remote `uv run`; it preserves the
all-extras environment prepared by the cluv launcher. `cluv-results` is a
remote symlink to the configured results path and is visible to all nodes on a
shared filesystem.

## Status and results

```bash
uv run scripts/cluv_prime_rl.py status
uv run scripts/cluv_prime_rl.py sync mila
```

The second command fetches remote results. With the default local
`SCRATCH=$HOME/scratch`, they are stored under
`$HOME/scratch/prime-values/<cluster>_<job-id>/`. Each directory contains the
resolved configs, component logs, rollouts, and Slurm output for that run.
Every `submit` and `sync` performs this fetch. The launcher excludes
`checkpoints/`, `broadcasts/`, and `weights/` directories, which can reach tens
of gigabytes per run; read them on the cluster under the configured
`results_path`.

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
| Mila | `mila` | `main` | 2 GPUs |
| Tamia | `aip-bengioy` | `gpubase_bynode_b2` | one 4× H100 node |
| Fir | `rrg-bengioy-ad_gpu` or `def-bengioy_gpu` | `gpubase_bygpu_b2` | 2× H100 |

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

## Submit a single-node experiment

Pass Slurm options before `--` and the prime-rl command after it. The defaults
request 16 CPUs, 128 GB of RAM, and four hours. GPU resources are
cluster-specific; command-line values override them.

```bash
uv run scripts/cluv_prime_rl.py submit mila \
  --gpus-per-node=8 --cpus-per-task=32 --mem=256G --time=12:00:00 \
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
  --gpus-per-node=1 --time=2:00:00 \
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
resolved configs, component logs, checkpoints, rollouts, and Slurm output for
that run.

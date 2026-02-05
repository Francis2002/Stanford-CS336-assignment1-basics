# Reproducibility Guide: Local & Remote (Cluster) Setup

This project uses **uv** for high-speed, reproducible dependency management. Follow these steps to set up your environment locally and on an internet-restricted GPU cluster.

---

## 1. Local Machine Setup

Ensure you have **uv** installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`).

1. **Clone the repository:**
```bash
git clone <repo-url>
cd <repo-name>

```


2. **Sync dependencies:**
```bash
uv sync

```



This automatically creates a `.venv` and installs the exact versions from `uv.lock`.


3. **Run code:**
```bash
uv run python <your_script.py>

```



---

## 2. Remote Cluster Setup (Internet-Restricted)

Since the cluster lacks internet access, you must build a portable environment locally and transfer it.

### Step A: Build the Portable Environment (On Local Machine)

Run these commands in your project root to create a standalone Linux-compatible environment:

```bash
# 1. Create a dedicated portable venv
uv venv .venv_cluster --link-mode copy

# 2. Populate it with Linux/CUDA dependencies
UV_PROJECT_ENVIRONMENT=.venv_cluster uv sync --link-mode copy

```

### Step B: Package and Transfer

1. **Compress the project:**
```bash
zip -r cluster_dist.zip . -x ".venv/*" -x ".git/*"

```


*(Note: This excludes your local symlinked `.venv` but includes the portable `.venv_cluster`.)*
2. **SCP to the cluster:**
```bash
scp cluster_dist.zip user@cluster-address:~/path/to/project/

```



### Step C: Execute on Cluster

1. **Unpack:**
```bash
unzip cluster_dist.zip
cd <repo-name>

```


2. **Run (Direct Binary Method):**
The safest way to run code is to point directly to the python binary in the cluster venv. This avoids path issues:
```bash
./.venv_cluster/bin/python <your_script.py>

```



---

## 3. Maintenance & Incremental Updates

After the initial 6GB transfer, **do not resend the big zip.** Use `rsync` from your local machine to sync only your code changes in seconds:

```bash
# Update remote code without touching the heavy venv
rsync -avz --exclude '.venv*' --exclude '.git' ./ user@cluster-address:~/path/to/project/

```

## Environment Details

* 
**Python:** 3.12+ 


* 
**PyTorch:** 2.6.0+cu124 (Linux) 
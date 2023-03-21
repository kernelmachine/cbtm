#!/usr/bin/env python3

import os
import subprocess

PIP_MSG = "Missing required package: pip install azure-storage-blob timeout-decorator more_itertools tokenizers"

on_aws = os.path.exists("/fsx")
on_faircluster = os.path.exists("/checkpoints")
on_azure = os.path.exists("/sched/cyclecloud.conf")
INSTRUCTIONS = []
try:
    import more_itertools
    import timeout_decorator
    import tokenizers

    if on_azure:
        from azure.storage.blob import BlobServiceClient
except ImportError:
    INSTRUCTIONS.append(PIP_MSG)

try:
    import megatron
except ImportError:
    INSTRUCTIONS.append(
        """
    Missing required package. run:

        git clone --branch fairseq_v2 git@github.com:ngoyal2707/Megatron-LM.git
        cd Megatron-LM
        pip install -e .
        cd ..
    """
    )

try:
    import fairscale
except ImportError:
    INSTRUCTIONS.append(
        "Missing required package. Do the install instructions (build from source) at https://github.com/facebookresearch/fairscale/blob/main/docs/source/installation_instructions.rst"
    )


if on_azure and not os.environ.get("AZURE_BLOB_SAS_URL", "").startswith(
    "https://fairacceleastus.blob.core.windows.net/"
):
    # We require checkpoints to be written to blob storage in the East US region. Please update your env variable to
    INSTRUCTIONS.append(
        """
        export AZURE_BLOB_SAS_URL="https://fairacceleastus.blob.core.windows.net/?sv=2020-08-04&ss=b&srt=sco&sp=rwdlactfx&se=2023-10-06T11:23:33Z&st=2021-10-06T03:23:33Z&spr=https&sig=s6aw4Ca4Ohbr7LQ%2BG9s58PEyYJsbXHjs%2Fc%2BuoTvzTUo%3D"
        """
    )


### Make sure correct tag/branch is used. Following consts define these expectations.
FAIRSCALE_HASH = "1bc96fa8c69def6d990e42bfbd75f86146ce29bd"
MEGATRON_BRANCH = "fairseq_v2"

# Check fairscale
fairscale_git_path = str(fairscale.__file__).replace("fairscale/__init__.py", ".git")
p_fairscale = subprocess.Popen(
    ["git", "--git-dir", fairscale_git_path, "rev-parse"],
    stdout=subprocess.PIPE,
)
branch = p_fairscale.communicate()[0].decode("utf-8").strip()
if branch != FAIRSCALE_HASH:
    INSTRUCTIONS.append(
        f"cd {fairscale_git_path}; git checkout {FAIRSCALE_HASH}`; cd -"
    )

# Check megatron
megatron_git_path = str(megatron.__file__).replace("megatron/__init__.py", ".git")
p_megatron = subprocess.Popen(
    ["git", "--git-dir", megatron_git_path, "symbolic-ref", "--short", "HEAD"],
    stdout=subprocess.PIPE,
)
branch = p_megatron.communicate()[0].decode("utf-8").strip()
if branch != MEGATRON_BRANCH:
    raise RuntimeError(
        f"Wrong megatron branch. Got `{branch}`, should be `{MEGATRON_BRANCH}`. Please run `git checkout {MEGATRON_BRANCH}` in its directory"
    )

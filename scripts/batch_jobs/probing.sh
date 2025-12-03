#!/bin/sh
# PBS -q rt_HF
# PBS -l select=1
# PBS -l walltime=10:00:00
# PBS -P gch51615

export PYENV_ROOT="$HOME/.pyenv" && export PATH="$PYENV_ROOT/bin:$PATH" && eval "$(pyenv init --path)" && eval "$(pyenv init -)"
pyenv activate vissl

cd ${PBS_O_WORKDIR}

# source /etc/profile.d/modules.sh
# module load cuda/12.6/12.6.1

bash scripts/probing.sh > probing_ijepa_vitb_in1k.log 2>&1
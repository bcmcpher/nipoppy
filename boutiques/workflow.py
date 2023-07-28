
import os
from pathlib import Path
import argparse
import json

from boutiques.puller import Puller

HELPTEXT = """
Import / cache the zenodo IDs for the processes installed by the user.
"""

## build all the paths
parser = argparse.ArgumentParser(description=HELPTEXT)
parser.add_argument('--global_config', type=str, required=True, help='path to global config file for your mr_proc dataset')
parser.add_arbument('--zenodo_id', type=str, required=True, help='a valid zenodo id to be cached in the project directory')
args = parser.parse_args()

## load global config
global_config_file = args.global_config
with open(global_config_file, 'r') as f:
    global_configs = json.load(f)

## user facing function to add zenodo IDs to global?
## ... or should those be tracked separately?

import boutiques
from boutiques import bosh
from boutiques.descriptor2func import function

## force to run in singularity (still old name...)

## test working directory to figure out where stuff can be cached
os.chdir('/home/bcmcpher/Projects/dl-boutiques/btq-direct')

## most basic run
out = bosh(["example.json", "zenodo.1482743"])
print(out)

## run advanced run
fslbet = function("zenodo.1482743")
out = fslbet(infile="./input/T1w.nii.gz", maskfile="./output/mask.nii.gz")

print(out)

##
## cache a zenodo ID - a .json file in ~/.cache/boutiques/production
##

x = boutiques.pull("zenodo.3240521")

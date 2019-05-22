# Author: Arthur Joe-Leun Tham (thamaj)

import tarfile
from pathlib import Path
import re

TEST = False
VIDEO_DIR = "./video/"
DATA = dict()
# {iteration: {steps,reward}}
MAP = [DATA]


path_list = Path(VIDEO_DIR).glob('**/*.tgz')

for path in sorted(path_list, key=lambda x: str(x)):
    path_str = str(path)
    #print(path_str)
    iteration_id = int(re.search(r'\d+', path_str.split(sep="-")[-1]).group())
    map_id = int(re.search(r'\d+', path_str.split(sep="-")[-2]).group())

    tarfile_file = tarfile.open(name=path_str, mode="r:gz")
    """tarfile_rewards = None
    for entry in tarfile_file:
        print(entry.name)
        if "rewards.txt" in entry.name:
            tarfile_rewards = tarfile_file.extractfile(entry)
            break
    if tarfile_rewards is None:
        raise Exception()"""
    tarfile_rewards = tarfile_file.extractfile(
        [entry for entry in tarfile_file if "rewards.txt" in entry.name][0])
    tarfile_rewards_lines = [str(line).rstrip('\n') for line in tarfile_rewards]

    MAP[map_id][str(iteration_id)] = {
        "steps" : len(tarfile_rewards_lines),
        "reward": int(re.search(r'[-]?\d+', tarfile_rewards_lines[-1].split(sep=":")[-1]).group())
    }
    tarfile_file.close()

    if TEST:
        break

for i in range(len(MAP)):
    print("MAP "+str(i))
    for j in sorted(MAP[i].items(),key=lambda x:int(x[0])):
        print(j)
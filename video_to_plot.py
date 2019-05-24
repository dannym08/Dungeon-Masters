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

RESULT = "map,iteration,reward,steps\n"

for i in range(len(MAP)):
    print("MAP "+str(i))
    for j,k in sorted(MAP[i].items(),key=lambda x:int(x[0])):
        print(j,k)
        k = list(k.values())
        RESULT += ""+str(i)+","+str(j)+","+str(k[0])+","+str(k[1])+"\n"

#print(RESULT)

file = open("video_to_plot.csv", "w")
file.write(RESULT)
file.close()
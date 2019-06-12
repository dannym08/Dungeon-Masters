# Author: Arthur Joe-Leun Tham (thamaj)

import tarfile
from pathlib import Path
import re
import sys

TEST = False
VIDEO_DIR = "./video/"
DATA = dict()
# {iteration: {steps,reward}}
MAP = [DATA]
COUNTER_MAX = 150000
COUNTER = 0
COUNTER_STEP = 1

path_list = Path(VIDEO_DIR).glob('**/*.tgz')

if __name__ == "__main__":
    #print(sys.argv)
    if len(sys.argv) == 2:
        COUNTER_STEP = int(sys.argv[1])
        #print(COUNTER_STEP)
    try:
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
                "reward": sum([
                    int(
                        re.search(r'[-]?\d+', _i.split(sep=":")[-1]).group()
                    ) for _i in tarfile_rewards_lines
                ])
            }
            tarfile_file.close()

            if TEST or COUNTER >= COUNTER_MAX:
                break
            COUNTER += 1

        RESULT = "map,iteration,steps,reward\n"

        for i in range(len(MAP)):
            print("MAP "+str(i))
            for j,k in sorted(MAP[i].items(),key=lambda x:int(x[0])):
                #print(int(j) % COUNTER_STEP)
                if COUNTER_STEP > 1 and (int(j) % COUNTER_STEP) != 0:
                    continue
                print(j, k)
                k = list(k.values())
                RESULT += ""+str(i)+","+str(j)+","+str(k[0])+","+str(k[1])+"\n"

        #print(RESULT)

        file = open("video_to_plot.csv", "w")
        file.write(RESULT)
        file.close()
    except Exception as e:
        print("An exception has occurred!")
        raise e
    finally:
        input("Press enter to close...")
import glob
from subprocess import run

def GetCheckpointIt(filepath):
"""Get the iteration number of a checkpoint from it's filename"""
    try:
        index_current = filepath.index("it_") + 3
    except:
        return -1
    done = False
    output_string = ""
    while done == False:
        output_string += filepath[index_current]
        index_current += 1
        if filepath[index_current] == ".":
            done = True
        
    return int(output_string)

simulation_directory = "${simulationDirectory}"
data_subdir = "${dataSubfolder}"

directory_list = glob.glob(simulation_directory + "output-[0-9][0-9][0-9][0-9]/" + data_subdir)

if not(directory_list):
    print("No directories found")

for directory in sorted(directory_list):
    print("Checking " + directory)
    checkpoint_list = glob.glob(directory + "checkpoint.chkpt.it_*")
    checkpoints = []
    for checkpoint in checkpoint_list:
        checkpoint_it = int(GetCheckpointIt(checkpoint))
        if not(checkpoint_it in checkpoints):
            checkpoints.append(checkpoint_it)
    if not(checkpoints):
        print("No Checkpoints found in " + directory)
        continue
    checkpoint_to_keep = max(checkpoints)
    print("Found " + str(len(checkpoints)) + " checkpoint(s)")
    for checkpoint in sorted(checkpoint_list):
        if not("_" + str(checkpoint_to_keep) + "." in checkpoint):
            print("Deleting " + checkpoint)
            run(["rm",checkpoint])
        else:
            print("Keeping checkpoint " + str(checkpoint_to_keep))
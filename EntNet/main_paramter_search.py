import time
import os
import numpy as np

# Hyperparameter nr 1
hyper1_name = 'Learning_Rate'
hyper1 = [1e-4, 1e-2]

# Hyperparameter nr 2
hyper2_name = 'Batchsize'
hyper2 = [32]

# Define task specific parameters
task = 1
epochs = 50
input_path = '/storage/remote/atcremers45/s0238/en-10k/'

exec_path = os.getcwd() + '/__main__.py'

for hp1 in hyper1:
    for hp2 in hyper2:
        execute = '{exe} --babi_task {t} --dataset_path {dir} --epochs {ep} --save_interval 10 --lr {hp_1} --batchsize {hp_2}'.format(
            exe=exec_path, t=task, dir=input_path, ep=epochs, hp_1=hp1, hp_2=hp2)
        screen_name = '{n1}_{val1}_{n2}_{val2}'.format(n1=hyper1_name, n2=hyper2_name, val1=round(hp1, 2), val2=round(hp2, 2))
        os.system(f"screen -S {screen_name} -dm bash -c 'source ~/env/bin/activate;python3 {execute};exec sh'")
        time.sleep(60)  # Give it a minute to start so the server won't brake down

        # for testing: this command creates the file test.txt when openning the screen
        # os.system(f"screen -S screen-test -dm bash -c 'touch test.txt;exec sh'")

os.system("screen -list")  # Provides the user with a list of created screens

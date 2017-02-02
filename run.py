import os

tag = 'celebA_dcgan'
dataset = 'celebA'

command = 'python main.py --dataset %s --is_train True ' \
          '--sample_dir samples_%s --checkpoint_dir checkpoint_%s --tensorboard_run %s '%(dataset, tag, tag, tag)


os.system(command)

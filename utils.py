import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
import os 
import glob

tf.app.flags.DEFINE_integer('select_model',-1,'select model')
FLAGS = tf.app.flags.FLAGS

def prepare_checkpoint_path(save_path,restore):
    if not tf.gfile.Exists(save_path):
        tf.gfile.MkDir(save_path)
    else:
        if not restore:
            print("This directory is already exist")
            exit()
def configure_learning_rate(learning_rate_init_value,global_step,decay_steps,decay_rate,name):
    learning_rate = tf.train.exponential_decay(
        learning_rate_init_value,global_step,decay_steps,decay_rate,staircase=True)
    return learning_rate

def configure_optimizer(learning_rate):
    return tf.train.AdamOptimizer(learning_rate)

def get_last_ckpt_path(folder_path):
    meta_paths = sorted(glob.glob(os.path.join(folder_path,'*meta')))
    numbers=[]
    for meta_path in meta_paths:
        numbers.append(int(meta_path.split('-')[-1].split('.')[0]))

    numbers=np.asarray(numbers)
    sorted_idx = np.argsort(numbers)
    print(FLAGS.select_model)
    if FLAGS.select_model!= -1:
        latest_meta_path = meta_paths[sorted_idx[-1]]
        letest_meta_path = latest_meta_path.replace(
            latest_meta_path.split('-')[-1].split('.')[0],str(FLAGS.select_model))
        print(latest_meta_path)
    else:
        latest_meta_path = meta_paths[sorted_idx[-1]]

    ckpt_path = latest_meta_path.replace('.meta','')
    print(ckpt_path)
    return ckpt_path

def get_image_paths(image_folder):
    print(image_folder)
    possible_image_type = ['jpg','png','JPEG','jpeg']
    image_list = [image_path for image_paths 
    in [glob.glob(os.path.join(image_folder, '*.%s' % ext)) 
    for ext in possible_image_type] for image_path in image_paths]
    return image_list
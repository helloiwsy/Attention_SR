import os
import time
import glob
import cv2
import random
import numpy as np
import tensorflow as tf
import scipy.io as sio
import time
try:
    import data_util
except ImportError:
    from dataset import data_util

FLAGS = tf.app.flags.FLAGS
#./Your/Path/train_HR/*_HR.mat
def load_image(im_fn, hr_size):
    #Get the path of LR images
    low_dir = os.path.dirname(im_fn).replace('train_HR','train_LR_x'+str(FLAGS.sr_scale))
    hr_dir = os.path.dirname(im_fn)
    file_name = os.path.basename(im_fn).split('_HR')[0]

    #Paths of HR, HRHF, HRLF, LR
    hr_path = im_fn
    hrhf_path = os.path.join(hr_dir,file_name+'_HRHF.mat')
    hrlf_path = os.path.join(hr_dir,file_name+'_HRLF.mat')
    lr_path = os.path.join(low_dir,file_name+'_LR.mat')

    #Load the image matrix from the paths
    original_image  = sio.loadmat(hr_path)['HR']
    hrhf_image  = sio.loadmat(hrhf_path)['HRHF']
    hrlf_image  = sio.loadmat(hrlf_path)['HRLF']
    lr_image  = sio.loadmat(lr_path)['LR']

    #Choice the start point for cropping images Randomly
    h, w, _ = original_image.shape
    h_edge = h - hr_size
    w_edge = w - hr_size
    h_start = (np.random.randint(low=0, high=h_edge//int(FLAGS.sr_scale), size=1)[0])*int(FLAGS.sr_scale)
    w_start = (np.random.randint(low=0, high=w_edge//int(FLAGS.sr_scale), size=1)[0])*int(FLAGS.sr_scale)
    
    #Calculate relative position of LR from HR point
    lr_h_start = h_start // int(FLAGS.sr_scale)
    lr_w_start = w_start // int(FLAGS.sr_scale)
    lr_size = hr_size // int(FLAGS.sr_scale)    

    #Crop images 
    original_image_hr = original_image[h_start:h_start+hr_size, w_start:w_start+hr_size, :]
    hrhf_image_hr= hrhf_image[h_start:h_start+hr_size, w_start:w_start+hr_size, :]
    hrlf_image_hr= hrlf_image[h_start:h_start+hr_size, w_start:w_start+hr_size, :]
    lr_image_hr= lr_image[lr_h_start:lr_h_start+lr_size, lr_w_start:lr_w_start+lr_size, :]

    #Data agumentation
    if FLAGS.data==True:
        rand_num = np.random.randint(0,8)
        if rand_num<4:
            original_image_hr = np.rot90(original_image_hr,rand_num)
            hrhf_image_hr = np.rot90(hrhf_image_hr,rand_num)
            hrlf_image_hr = np.rot90(hrlf_image_hr,rand_num)
            lr_image_hr = np.rot90(lr_image_hr,rand_num)
        else:
            original_image_hr = np.flipud(np.rot90(original_image_hr,rand_num))
            hrhf_image_hr = np.flipud(np.rot90(hrhf_image_hr,rand_num))
            hrlf_image_hr = np.flipud(np.rot90(hrlf_image_hr,rand_num))
            lr_image_hr = np.flipud(np.rot90(lr_image_hr,rand_num))

    #Return images
    return original_image_hr,lr_image_hr,hrhf_image_hr,hrlf_image_hr

def get_record(image_path):
    original_path = glob.glob(image_path)    
    print('%d files found' % (len(original_path)))
    if len(original_path) == 0:
        raise FileNotFoundError('check your training dataset path')
    index = list(range(len(original_path)))
    
    while True:
        random.shuffle(original_path)
        for i in index:
            im_fn = original_path[i]
            yield im_fn


def generator(image_path,hr_size=512, batch_size=32):
    hr_list,lr_list,hf_list,lf_list = [],[],[],[]
    for im_fn in get_record(image_path):
        try:
            o_hr,o_lr,hf_hr,lf_hr= load_image(im_fn, hr_size)
            hr_list.append(o_hr)
            lr_list.append(o_lr)
            hf_list.append(hf_hr)
            lf_list.append(lf_hr)
            
            if len(hr_list) == batch_size:
                yield hr_list,lr_list,hf_list,lf_list
                hr_list,lr_list,hf_list,lf_list = [],[],[],[]
        except FileNotFoundError as e:
            print(e)
            break
        except Exception as e:
            import traceback
            traceback.print_exc()
            continue

def get_generator(image_path,  **kwargs):
    return generator(image_path,  **kwargs)


def get_batch(image_path, num_workers, **kwargs):
    try:
        
        generator = get_generator(image_path, **kwargs)
        enqueuer = data_util.GeneratorEnqueuer(generator, use_multiprocessing=True)
        enqueuer.start(max_queue_size=24, workers=num_workers)
        generator_ouptut = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.001)
            yield generator_output
            generator_output = None

    finally:
        if enqueuer is not None:
            enqueuer.stop()



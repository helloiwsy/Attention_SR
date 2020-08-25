
import os 
import glob
import sys
import numpy as np
import tensorflow as tf 
import DSR
import utils
import cv2
import scipy.io as sio
import time

tf.app.flags.DEFINE_string('model_path', '/where/your/model/folder', '')
tf.app.flags.DEFINE_string('run_gpu', '0', '')
tf.app.flags.DEFINE_integer('sr_scale',2,'Super resolutio scale')
FLAGS = tf.app.flags.FLAGS

def load_model(model_path):
    input_low_images = tf.placeholder(tf.float32, shape=[1,None,None,3],name = 'input_low_images')
    model_builder = DSR.Model_Build()
    main_hf,main_lf= model_builder.generator(input_low_images)
    main_out_clip = tf.cast(tf.round(tf.clip_by_value(main_hf+main_lf,0,255)),tf.uint8)
    all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    gen_vars = [var for var in all_vars if var.name.startswith('generator')]
    saver = tf.train.Saver(gen_vars)
    sess = tf.Session(config = tf.ConfigProto(allow_soft_placement=True))
    ckpt_path = utils.get_last_ckpt_path(model_path)
    print("model_ path : ",ckpt_path)
    saver.restore(sess,ckpt_path)

    return input_low_images,main_out_clip,main_hf,main_lf,sess


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.run_gpu
    input_image,main_output,hf_output,lf_output,sess = load_model(FLAGS.model_path)
    
    #######################################
    #Where is your test low-resolution images
    ########################################
    test_image_path = './Test/image/path'

    output_path = ''
    test_image_list = utils.get_image_paths(test_image_path)
    temp_time = 0 

    for test_idx,test_image in enumerate(test_image_list):
        loaded_image = cv2.imread(test_image)
        feed_dict ={input_image:[loaded_image[:,:,::-1]]}
        
        main_out,hf_out,lf_out = sess.run([main_output,hf_output,lf_output]feed_dict=feed_dict)

        main_image = main_out[0,:,:,:]
        hf_image = hf_out[0,:,:,:]
        lf_image = lf_out[0,:,:,:]
        image_name = os.path.basename(test_image).split('.')[0]
        ################################
        #Where to Save the output images
        ################################
        #For examples
        
        #cv2.imwrite('Your main output path'+image_name+'.png',main_image[:,:,::-1])
        #cv2.imwrite('Your hf output path'+image_name+'.png',hf_image[:,:,::-1])
        #cv2.imwrite('Your lf output path'+image_name+'.png',lf_image[:,:,::-1])
        
        #If you need full negative value of high frequnecy(hf_image)
        #you can use sio.savemat() and check the document page.

       
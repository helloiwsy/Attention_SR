import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
import cv2
from dataset import SR_data_load
import DSR  
import loss
import utils
import glob
import scipy.io as sio  


tf.app.flags.DEFINE_string('run_gpu','0','use single gpu')
tf.app.flags.DEFINE_string('m','default','Comment about this model')
tf.app.flags.DEFINE_string('save_path','/where/your/folder','')
tf.app.flags.DEFINE_boolean('model_restore',False,'')
tf.app.flags.DEFINE_string('hr_path','./train_HR/*_HR.mat','Where is  Data Imageset')
tf.app.flags.DEFINE_integer('batch_size',16,'')
tf.app.flags.DEFINE_integer('input_size',32,'')
tf.app.flags.DEFINE_float('alpha',0.1,'')
tf.app.flags.DEFINE_float('decay_rate',0.5,'')
tf.app.flags.DEFINE_integer('decay_steps',300000,'')
tf.app.flags.DEFINE_integer('num_workers',12,'')
tf.app.flags.DEFINE_integer('max_to_keep',10,'how many do you wnat to save model?')
tf.app.flags.DEFINE_integer('save_model_steps',10000,'')
tf.app.flags.DEFINE_integer('save_summary_steps',100,'')
tf.app.flags.DEFINE_integer('max_steps',4000000,'')
tf.app.flags.DEFINE_integer('sr_scale',2,'Super resolution scale')
tf.app.flags.DEFINE_boolean('data',True,'Data augmentation')

tf.app.flags.DEFINE_float('learning_rate',0.0001,'define your lr')

FLAGS = tf.app.flags.FLAGS

def human_format(num, round_to=3):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num = round(num / 1000.0, round_to)
    return '{:.{}f}{}'.format(round(num, round_to), round_to, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

def main(argv=None):
    os.environ['CUDA_VISIBLE_DEVICES']=FLAGS.run_gpu
    utils.prepare_checkpoint_path(FLAGS.save_path,FLAGS.model_restore)

    input_size = FLAGS.input_size  # 32
    high_size = int(FLAGS.input_size * FLAGS.sr_scale)
    
    with tf.name_scope('Input_Placeholders'):
        input_original_lr = tf.placeholder(
                tf.float32,shape=[FLAGS.batch_size,input_size,input_size,3],name="Input_Original_LR")  
        input_original_hr = tf.placeholder(
                tf.float32,shape=[FLAGS.batch_size,high_size,high_size,3],name="Input_Original_HR")       
        input_hf_hr= tf.placeholder(
                tf.float32,shape=[FLAGS.batch_size,high_size,high_size,3],name="Input_HF_HR")
        input_lf_hr= tf.placeholder(
                tf.float32,shape=[FLAGS.batch_size,high_size,high_size,3],name="Input_LF_HR")
    
    model_builder  = DSR.Model_Build()
   
    gen_hf, gen_lf = model_builder.generator(input_original_lr)
    cast_val_original = tf.cast(tf.round(tf.clip_by_value(val_hf+val_lf,0,255)),tf.uint8,name='Validation_Original_SR')
    
    decom_sr_loss =0.0
    loss_weights = tf.Variable([1.,1.],trainable=True,name="Lossweight")
    loss_val_init=tf.Variable([1.,1.],name='Loss_init_Value')
    
    with tf.name_scope('Several_Losses'):

        loss_lf = tf.reduce_mean(tf.abs(input_lf_hr - gen_lf))
        loss_hf = tf.reduce_mean(tf.abs(input_hf_hr - gen_hf))


        loss_lf_weighted = loss_lf*loss_weights[0]
        loss_hf_weighted = loss_hf*loss_weights[1]
        decom_sr_loss = loss_lf_weighted  + loss_hf_weighted

    global_step = tf.get_variable('global_step',[],dtype=tf.int64,initializer=tf.constant_initializer(0),trainable=False)
    train_vars = tf.trainable_variables()
    learing_rate = utils.configure_learning_rate(FLAGS.learning_rate,global_step,FLAGS.decay_steps,FLAGS.decay_rate,'High_Gener')
    
    with tf.name_scope('Optimizer'):
        generator_vars =[var for var in train_vars if var.name.startswith('generator')]
        last_conv_vars =[var for var in train_vars if var.name.startswith('generator/Decomposition/decom_last_layer')]
        loss_weight_var = [var for var in train_vars if var.name.startswith('Lossweight')]
        
        reassign_l1 = loss_val_init[0].assign(loss_lf)
        reassign_p1 = loss_val_init[1].assign(loss_hf)
        
        gen_optimizer = utils.configure_optimizer(learing_rate)

        conv_grad_lf_l1=gen_optimizer.compute_gradients(loss_lf_weighted,var_list=last_conv_vars)
        conv_grad_hf_l1=gen_optimizer.compute_gradients(loss_hf_weighted,var_list=last_conv_vars)
    
        loss_hat= [loss_lf/loss_val_init[0] ,loss_hf/loss_val_init[1]]
        loss_ratio = loss_hat/tf.reduce_mean(loss_hat)
        
        conv_gradients= [tf.norm(conv_grad_lf_l1[0]),tf.norm(conv_grad_hf_l1[0])]
        conv_gradients_mean= tf.reduce_mean(conv_gradients)
        
        diff = tf.abs(conv_gradients - (loss_ratio**FLAGS.alpha)*conv_gradients_mean)
        grad_norm_loss=tf.reduce_sum(diff)

        loss_weight_gradients=gen_optimizer.compute_gradients(grad_norm_loss,var_list=loss_weight_var)
        gen_gradients=gen_optimizer.compute_gradients(decom_sr_loss,var_list=generator_vars)

        loss_weight_update = gen_optimizer.apply_gradients(loss_weight_gradients)
        gen_grad_updates = gen_optimizer.apply_gradients(gen_gradients,global_step=global_step) 

       
        with tf.control_dependencies([loss_weight_update,gen_grad_updates]):
            Loss_weights_changed= (loss_weights*(2/tf.reduce_sum(loss_weights)))
            loss_update_a = loss_weights[0].assign(Loss_weights_changed[0])
            loss_update_b = loss_weights[1].assign(Loss_weights_changed[1])         
            with tf.control_dependencies([loss_update_a,loss_update_b]):
                train_op = tf.no_op(name='train_op')             
    saver = tf.train.Saver(max_to_keep=FLAGS.max_to_keep)

    data_generator = SR_data_load.get_batch(image_path=FLAGS.hr_path,num_workers = FLAGS.num_workers,batch_size=FLAGS.batch_size, hr_size=high_size)
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(FLAGS.save_path,tf.get_default_graph())
    time_for_one_batch=0

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True))as sess:
        if FLAGS.model_restore:
            ckpt  = tf.train.latest_checkpoint(FLAGS.save_path)
            saver.restore(sess,ckpt)
        else:
            sess.run(tf.global_variables_initializer())

        start_time = time.time()
        print("Start ! Training!!")
        for iter_val in range(int(global_step.eval())+1,FLAGS.max_steps+1):
            batch_time= time.time()
            data = next(data_generator)
            time_for_one_batch =time_for_one_batch+ time.time()-batch_time
            original_hr,original_lr,hf_image,lf_image =\
            np.asarray(data[0]),np.asarray(data[1]),np.asarray(data[2]),np.asarray(data[3])
            feed_dict={
                input_original_hr : original_hr,
                input_original_lr : original_lr,
                input_hf_hr:hf_image,
                input_lf_hr:lf_image }

            if iter_val == 1 :    
                sess.run([reassign_l1,reassign_p1],feed_dict=feed_dict)
            _ = sess.run(train _op,feed_dict=feed_dict)
            
            if iter_val!=0 and iter_val % FLAGS.save_summary_steps==0:
                loss_lf_val,loss_hf_val,loss_decom =\
                    sess.run([loss_lf,loss_hf,decom_sr_loss],feed_dict=feed_dict)
                w_loss_lf_val,w_loss_hf_val,l_w =sess.run([loss_lf_weighted,loss_hf_weighted,loss_weights],feed_dict=feed_dict)

                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, global_step=iter_val)
                used_time = time.time() - start_time
                avg_time_per_step = used_time / FLAGS.save_summary_steps
                avg_examples_per_second = (FLAGS.save_summary_steps * FLAGS.batch_size) / used_time 
                
                print("-------------------------------------------------------------")
                print("[Experiment Comment:][%s]"%(FLAGS.m))
                print('[S:%s][%.2fsample/sec|%.2fsec/step|%.3fsec/batch|Data_aug:%s]'%\
                    (human_format(iter_val),avg_examples_per_second,avg_time_per_step,time_for_one_batch,FLAGS.data))            
                print('<Weighted Loss : %.4f = W_LF:%.4f + W_HF:%.4f >'%(loss_decom,w_loss_lf_val,w_loss_hf_val))
                print('<Original Loss | LF :%.4f | HF :%.4f    >'%(loss_lf_val,loss_hf_val))
                print('<Weights of loss : ',l_w,'>')
                time_for_one_batch=0
                start_time=time.time()
            if iter_val != 0 and iter_val % FLAGS.save_model_steps == 0:
                        checkpoint_fn = os.path.join(FLAGS.save_path, 'model.ckpt'+str(iter_val/FLAGS.save_model_steps)+'_'+str(last_psnr[0]))
                        saver.save(sess, checkpoint_fn, global_step=iter_val)


                

    print('')
    print('*' * 30)
    print(' Training done!!! ')
    print('*' * 30)
    print('')

if __name__ == '__main__':
    tf.app.run()

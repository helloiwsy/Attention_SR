import tensorflow as tf 
import numpy as np
from tensorflow import layers
from tensorflow.contrib  import slim

FLAGS = tf.app.flags.FLAGS

class Model_Build:
    def __init__(self):
        return

    def preprocess(self,images):
        with tf.variable_scope("Preprocessing"):
            return (images/127.5) - 1.0

    def postprocess(self,images):
        with tf.variable_scope("Postprocessing"):
            return (images+1.0)*127.5 
    #################
    #Upsampler Module
    #################

    def phaseShift(self,features,scale,shape_1,shape_2):
        X = tf.reshape(features,shape_1)
        X = tf.transpose(X,[0,1,3,2,4])
        return tf.reshape(X,shape_2)

    def pixelShuffler(self, features, scale=2):
        with tf.variable_scope('pixel_shuffler'):
            size = tf.shape(features) # features = [n, h, w, c]
            batch_size = size[0]
            h, w, c= size[1], size[2], features.get_shape().as_list()[-1]# 64
            channel_target = c//(scale*scale)  # 16
            channel_factor = c//channel_target # 4
            shape_1 = [batch_size, h, w, channel_factor//scale, channel_factor//scale]
            shape_2 = [batch_size, h*scale,w*scale, 1] # [n, 2w, 2h, 1]
            input_split =  tf.split(axis=3, num_or_size_splits = channel_target, value=features)
            output = tf.concat([self.phaseShift(x, scale, shape_1, shape_2) for x in input_split],axis=3)
            return output
        
    def upsampler(self,features,scale=2):
        features = self.pixelShuffler(features,scale=2)
        if scale is 4 or scale is 8 :
            features =layers.conv2d(features,64,3,strides=1,padding='SAME',activation=None,name="upsample_x4_conv")
            features = self.pixelShuffler(features,scale=2)
        if scale is 8:
            features =layers.conv2d(features,64,3,strides=1,padding='SAME',activation=None,name="upsample_x8_conv")
            features = self.pixelShuffler(features,scale=2)
        return features

    ################
    #Reisdual Module
    ################
    def res_group(self,features,rb_num,conv_num,out_ch,res_scale,scope):
        input_features = features
        with tf.variable_scope(scope):
            for idx in range(rb_num):
                features = self.res_block(features,conv_num,out_ch,res_scale,scope='RB_%d'%(idx))
            features = layers.conv2d(features,out_ch,3,strides=1,padding='SAME',activation=None,name='Last_Conv') 
            return input_features+features

    def res_block(self,features,conv_num,out_ch,res_scale, scope):
        input_features = features
        with tf.variable_scope(scope):
            features = layers.conv2d(input_features,out_ch,3,strides=1,padding='SAME',activation=tf.nn.leaky_relu)
            for i in range(conv_num-2):
                features = layers.conv2d(features,out_ch,3,strides=1,padding='SAME',activation=tf.nn.leaky_relu)
            features = layers.conv2d(features,out_ch,3,strides=1,padding='SAME',activation=None) 
            
            #################################
            ###Insert Attention module here##
            #To Insert Attention module, Remove the comment (#) below line 
            #################################
            #features = self.channel_attention(features,4)
            #################################

            return input_features+features*res_scale
    
    ################## 
    #Attention Module
    ##################
    
    def channel_attention(self,feature,feat_n,ratio):
        input_feature = feature
        x = tf.reduce_mean(feature,axis=[1,2],keep_dims=True)
        x = slim.conv2d(x,feat_n//ratio,1,activation_fn=tf.nn.leaky_relu)
        x = slim.conv2d(x,feat_n,1,activation_fn=tf.nn.sigmoid)
        return tf.multiply(input_feature,x)
    
    def spatial_attention(self,feature):
        input_feature = feature
        x = tf.layers.conv2d(input_feature,1,1,strides=1,padding='SAME',activation=tf.nn.sigmoid)
        return tf.multiply(input_feature,x)
    
    #############
    #Main Network
    #############
    def generator(self,inputs):
        with tf.variable_scope('generator',reuse=tf.AUTO_REUSE):
            inputs = self.preprocess(inputs)
            with tf.variable_scope('Decomposition'):
                decom_input = layers.conv2d(inputs,64,3,padding='SAME',activation=tf.nn.leaky_relu,name='Decom_input')
                decom_feat =self.res_group(decom_input,10,3,64,0.1,scope="Decom_Res1")
                decom_out_feat = layers.conv2d(decom_feat,64,3,strides=1,padding='SAME',activation=None,name='decom_last_layer')

            with tf.variable_scope('SR'):
                with tf.variable_scope('LF'):
                    sr_lf_input = layers.conv2d(decom_out_feat,64,3,padding='SAME',activation=tf.nn.leaky_relu,name='LF_INPUT_FEATURE')

                    sr_lf_feat = self.res_group(sr_lf_input,10,2,64,0.1,scope="LF_RES1")
                    sr_lf_feat = self.res_group(sr_lf_feat,10,2,64,0.1,scope="LF_RES2")
                    sr_lf_feat = self.res_group(sr_lf_feat,10,2,64,0.1,scope="LF_RES3")
                    
                    sr_lf_feat = sr_lf_feat+sr_lf_input

                    upfeature_lf = self.upsampler(sr_lf_feat,int(FLAGS.sr_scale))
                    lf_hr_output = layers.conv2d(upfeature_lf,3,3,padding='SAME',name='Recon_LF')

                with tf.variable_scope('HF'):    
                    sr_hf_input = layers.conv2d(decom_out_feat,64,3,padding='SAME',activation=tf.nn.leaky_relu,name='HF_INPUT_FEATURE')

                    sr_hf_feat = self.res_group(sr_hf_input,10,2,64,0.1,scope="HF_RES1")
                    sr_hf_feat = self.res_group(sr_hf_feat,10,2,64,0.1,scope="HF_RES2")
                    sr_hf_feat = self.res_group(sr_hf_feat,10,2,64,0.1,scope="HF_RES3")
   
                    sr_hf_feat = sr_hf_feat+sr_hf_input

                    upfeature_hf = self.upsampler(sr_hf_feat,int(FLAGS.sr_scale))
                    hf_hr_output = layers.conv2d(upfeature_hf,3,3,padding='SAME',name='Recon_HF')
            
            lf_hr = self.postprocess(lf_hr_output)
            hf_hr = self.postprocess(hf_hr_output)
        return hf_hr , lf_hr
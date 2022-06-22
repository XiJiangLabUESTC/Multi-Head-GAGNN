from __future__ import division
from scipy.io import loadmat
import tensorflow as tf
from tqdm import tqdm
import os
import numpy as np
import nibabel as nib
import scipy.stats as stats
from scipy.stats import pearsonr
import scipy.io as scio
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def load_data():
    temp1 = []
    temp2 = []
    with open("/media/D/alex/MIA_MH_GAGNN/4DfMRI_input/make_fold/train1.txt", "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            index_id = int(line)
            train_path = '/media/D/alex/MIA_MH_GAGNN/4DfMRI_input/data_fMRI_48_56_48/%s/emotion/input_data.mat' % (index_id)
            train_data = loadmat(train_path)
            train_data = train_data['input_data']
            temp1.append(train_data)
            for rsn_id in range(10):
                label_path = '/media/D/alex/MIA_MH_GAGNN/4DfMRI_input/dl_result/%s/emotion/rsn%s_spatial.mat' % (index_id, rsn_id+1)
                label_data = loadmat(label_path)
                label_data = label_data['space']
                temp2.append(label_data)
    x = np.array(temp1)
    x = x.reshape(-1, 48, 56, 48, 176)  # x,y,z,t
    x = x.astype(np.float32)
    y_s = np.array(temp2)
    y_s = y_s.reshape(-1, 48, 56, 48)  # x,y,z
    y_s = y_s.astype(np.float32)

    return x, y_s


def Conv3d(x, k_size, s_size, c_in, c_out, pad_type, filter_name):
    filter_sets = tf.get_variable(filter_name, shape=[k_size, k_size, k_size, c_in, c_out], initializer=tf.keras.initializers.he_normal())
    c = tf.nn.conv3d(x, filter_sets, strides=[1, s_size, s_size, s_size, 1], padding=pad_type)
    bn = tf.keras.layers.BatchNormalization()
    c_temp = bn(c, training=True)
    output = tf.nn.relu(c_temp)
    return output


def Deconv3d(x, out1, out2, out3, c_in, c_out, filter_name):
    filter_sets = tf.get_variable(filter_name, shape=[2, 2, 2, c_out, c_in], initializer=tf.keras.initializers.he_normal())
    dc = tf.nn.conv3d_transpose(x, filter_sets, output_shape=[1, out1, out2, out3, c_out], strides=[1, 2, 2, 2, 1], padding='SAME')
    bn = tf.keras.layers.BatchNormalization()
    dc_temp = bn(dc, training=True)
    output = tf.nn.relu(dc_temp)
    return output


def GAT(x, weight, weight_q, weight_k):
    (b, d, h, w, c) = x.shape
    (b, d, h, w, c) = (int(b), int(d), int(h), int(w), int(c))
    temp1 = tf.reshape(Conv3d(x, 3, 1, c, c, 'SAME', weight_q), [d * h * w, c])
    temp2 = tf.reshape(Conv3d(x, 3, 1, c, c, 'SAME', weight_k), [d * h * w, c])
    temp_A = tf.matmul(temp1, tf.transpose(temp2, [1, 0]))
    A = (temp_A-tf.reduce_min(temp_A))/tf.reduce_max(temp_A)
    temp_x = tf.reshape(x,[-1,c])
    weights = tf.get_variable(weight, shape=[c, c], initializer=tf.keras.initializers.he_normal())
    temp_output = tf.reshape(tf.matmul(tf.matmul(A, temp_x), weights), [b, d, h, w, c])
    bn = tf.keras.layers.BatchNormalization()
    temp = bn(temp_output, training=True)
    output = tf.nn.relu(temp)
    return output


def Xinet(x):
    base_num=48
    conv1 = Conv3d(x, 3, 1, 176, base_num, 'SAME', 'W1')
    conv2 = Conv3d(conv1, 3, 1, base_num, base_num, 'SAME', 'W2')
    conv3 = Conv3d(conv2, 2, 2, base_num, 2*base_num, 'VALID', 'W3')
    conv4 = Conv3d(conv3, 3, 1, 2*base_num, 2*base_num, 'SAME', 'W4')
    conv5 = Conv3d(conv4, 2, 2, 2*base_num, 4*base_num, 'VALID', 'W5')
    conv6 = Conv3d(conv5, 3, 1, 4*base_num, 4*base_num, 'SAME', 'W6')
    attention = GAT(conv6, 'AW','AQ', 'AK')
    # rsn0
    attention0 = GAT(attention, 'AW0', 'Q0', 'K0')
    conv7_0 = Conv3d(attention0, 3, 1, 4*base_num, 4*base_num, 'SAME', 'W7_0')
    up1_0 = Deconv3d(conv7_0, 24, 28, 24, 4*base_num, 4*base_num, 'WUP1_0')
    copy1_0 = tf.concat([conv4, up1_0], 4)
    conv8_0 = Conv3d(copy1_0, 3, 1, 6*base_num, 2*base_num, 'SAME', 'W8_0')
    up2_0 = Deconv3d(conv8_0, 48, 56, 48, 2*base_num, 2*base_num, 'WUP2_0')
    copy2_0 = tf.concat([conv2, up2_0], 4)
    conv9_0 = Conv3d(copy2_0, 3, 1, 3*base_num, base_num, 'SAME', 'W9_0')
    conv10_0 = Conv3d(conv9_0, 3, 1, base_num, base_num/2, 'SAME', 'W10_0')
    conv11_0 = Conv3d(conv10_0, 3, 1, base_num/2, base_num/4, 'SAME', 'W11_0')
    temp_rsn0 = Conv3d(conv11_0, 3, 1, base_num/4, 1, 'SAME', 'W12_0')
    rsn0 = tf.reshape(temp_rsn0, [48, 56, 48], name='rsn0_result')
    # rsn1
    attention1 = GAT(attention, 'AW1', 'Q1', 'K1')
    conv7_1 = Conv3d(attention1, 3, 1, 4*base_num, 4*base_num, 'SAME', 'W7_1')
    up1_1 = Deconv3d(conv7_1, 24, 28, 24, 4*base_num, 4*base_num, 'WUP1_1')
    copy1_1 = tf.concat([conv4, up1_1], 4)
    conv8_1 = Conv3d(copy1_1, 3, 1, 6*base_num, 2*base_num, 'SAME', 'W8_1')
    up2_1 = Deconv3d(conv8_1, 48, 56, 48, 2*base_num, 2*base_num, 'WUP2_1')
    copy2_1 = tf.concat([conv2, up2_1], 4)
    conv9_1 = Conv3d(copy2_1, 3, 1, 3*base_num, base_num, 'SAME', 'W9_1')
    conv10_1 = Conv3d(conv9_1, 3, 1, base_num, base_num/2, 'SAME', 'W10_1')
    conv11_1 = Conv3d(conv10_1, 3, 1, base_num/2, base_num/4, 'SAME', 'W11_1')
    temp_rsn1 = Conv3d(conv11_1, 3, 1, base_num/4, 1, 'SAME', 'W12_1')
    rsn1 = tf.reshape(temp_rsn1, [48, 56, 48], name='rsn1_result')
    # rsn2
    attention2 = GAT(attention, 'AW2', 'Q2', 'K2')
    conv7_2 = Conv3d(attention2, 3, 1, 4*base_num, 4*base_num, 'SAME', 'W7_2')
    up1_2 = Deconv3d(conv7_2, 24, 28, 24, 4*base_num, 4*base_num, 'WUP1_2')
    copy1_2 = tf.concat([conv4, up1_2], 4)
    conv8_2 = Conv3d(copy1_2, 3, 1, 6*base_num, 2*base_num, 'SAME', 'W8_2')
    up2_2 = Deconv3d(conv8_2, 48, 56, 48, 2*base_num, 2*base_num, 'WUP2_2')
    copy2_2 = tf.concat([conv2, up2_2], 4)
    conv9_2 = Conv3d(copy2_2, 3, 1, 3*base_num, base_num, 'SAME', 'W9_2')
    conv10_2 = Conv3d(conv9_2, 3, 1, base_num, base_num/2, 'SAME', 'W10_2')
    conv11_2 = Conv3d(conv10_2, 3, 1, base_num/2, base_num/4, 'SAME', 'W11_2')
    temp_rsn2 = Conv3d(conv11_2, 3, 1, base_num/4, 1, 'SAME', 'W12_2')
    rsn2 = tf.reshape(temp_rsn2, [48, 56, 48], name='rsn2_result')
    # rsn3
    attention3 = GAT(attention, 'AW3', 'Q3', 'K3')
    conv7_3 = Conv3d(attention3, 3, 1, 4*base_num, 4*base_num, 'SAME', 'W7_3')
    up1_3 = Deconv3d(conv7_3, 24, 28, 24, 4*base_num, 4*base_num, 'WUP1_3')
    copy1_3 = tf.concat([conv4, up1_3], 4)
    conv8_3 = Conv3d(copy1_3, 3, 1, 6*base_num, 2*base_num, 'SAME', 'W8_3')
    up2_3 = Deconv3d(conv8_3, 48, 56, 48, 2*base_num, 2*base_num, 'WUP2_3')
    copy2_3 = tf.concat([conv2, up2_3], 4)
    conv9_3 = Conv3d(copy2_3, 3, 1, 3*base_num, base_num, 'SAME', 'W9_3')
    conv10_3 = Conv3d(conv9_3, 3, 1, base_num, base_num/2, 'SAME', 'W10_3')
    conv11_3 = Conv3d(conv10_3, 3, 1, base_num/2, base_num/4, 'SAME', 'W11_3')
    temp_rsn3 = Conv3d(conv11_3, 3, 1, base_num/4, 1, 'SAME', 'W12_3')
    rsn3 = tf.reshape(temp_rsn3, [48, 56, 48], name='rsn3_result')
    # rsn4
    attention4 = GAT(attention, 'AW4', 'Q4', 'K4')
    conv7_4 = Conv3d(attention4, 3, 1, 4*base_num, 4*base_num, 'SAME', 'W7_4')
    up1_4 = Deconv3d(conv7_4, 24, 28, 24, 4*base_num, 4*base_num, 'WUP1_4')
    copy1_4 = tf.concat([conv4, up1_4], 4)
    conv8_4 = Conv3d(copy1_4, 3, 1, 6*base_num, 2*base_num, 'SAME', 'W8_4')
    up2_4 = Deconv3d(conv8_4, 48, 56, 48, 2*base_num, 2*base_num, 'WUP2_4')
    copy2_4 = tf.concat([conv2, up2_4], 4)
    conv9_4 = Conv3d(copy2_4, 3, 1, 3*base_num, base_num, 'SAME', 'W9_4')
    conv10_4 = Conv3d(conv9_4, 3, 1, base_num, base_num/2, 'SAME', 'W10_4')
    conv11_4 = Conv3d(conv10_4, 3, 1, base_num/2, base_num/4, 'SAME', 'W11_4')
    temp_rsn4 = Conv3d(conv11_4, 3, 1, base_num/4, 1, 'SAME', 'W12_4')
    rsn4 = tf.reshape(temp_rsn4, [48, 56, 48], name='rsn4_result')
    # rsn5
    attention5 = GAT(attention, 'AW5', 'Q5', 'K5')
    conv7_5 = Conv3d(attention5, 3, 1, 4*base_num, 4*base_num, 'SAME', 'W7_5')
    up1_5 = Deconv3d(conv7_5, 24, 28, 24, 4*base_num, 4*base_num, 'WUP1_5')
    copy1_5 = tf.concat([conv4, up1_5], 4)
    conv8_5 = Conv3d(copy1_5, 3, 1, 6*base_num, 2*base_num, 'SAME', 'W8_5')
    up2_5 = Deconv3d(conv8_5, 48, 56, 48, 2*base_num, 2*base_num, 'WUP2_5')
    copy2_5 = tf.concat([conv2, up2_5], 4)
    conv9_5 = Conv3d(copy2_5, 3, 1, 3*base_num, base_num, 'SAME', 'W9_5')
    conv10_5 = Conv3d(conv9_5, 3, 1, base_num, base_num/2, 'SAME', 'W10_5')
    conv11_5 = Conv3d(conv10_5, 3, 1, base_num/2, base_num/4, 'SAME', 'W11_5')
    temp_rsn5 = Conv3d(conv11_5, 3, 1, base_num/4, 1, 'SAME', 'W12_5')
    rsn5 = tf.reshape(temp_rsn5, [48, 56, 48], name='rsn5_result')
    # rsn6
    attention6 = GAT(attention, 'AW6', 'Q6', 'K6')
    conv7_6 = Conv3d(attention6, 3, 1, 4*base_num, 4*base_num, 'SAME', 'W7_6')
    up1_6 = Deconv3d(conv7_6, 24, 28, 24, 4*base_num, 4*base_num, 'WUP1_6')
    copy1_6 = tf.concat([conv4, up1_6], 4)
    conv8_6 = Conv3d(copy1_6, 3, 1, 6*base_num, 2*base_num, 'SAME', 'W8_6')
    up2_6 = Deconv3d(conv8_6, 48, 56, 48, 2*base_num, 2*base_num, 'WUP2_6')
    copy2_6 = tf.concat([conv2, up2_6], 4)
    conv9_6 = Conv3d(copy2_6, 3, 1, 3*base_num, base_num, 'SAME', 'W9_6')
    conv10_6 = Conv3d(conv9_6, 3, 1, base_num, base_num/2, 'SAME', 'W10_6')
    conv11_6 = Conv3d(conv10_6, 3, 1, base_num/2, base_num/4, 'SAME', 'W11_6')
    temp_rsn6 = Conv3d(conv11_6, 3, 1, base_num/4, 1, 'SAME', 'W12_6')
    rsn6 = tf.reshape(temp_rsn6, [48, 56, 48], name='rsn6_result')
    # rsn7
    attention7 = GAT(attention, 'AW7', 'Q7', 'K7')
    conv7_7 = Conv3d(attention7, 3, 1, 4*base_num, 4*base_num, 'SAME', 'W7_7')
    up1_7 = Deconv3d(conv7_7, 24, 28, 24, 4*base_num, 4*base_num, 'WUP1_7')
    copy1_7 = tf.concat([conv4, up1_7], 4)
    conv8_7 = Conv3d(copy1_7, 3, 1, 6*base_num, 2*base_num, 'SAME', 'W8_7')
    up2_7 = Deconv3d(conv8_7, 48, 56, 48, 2*base_num, 2*base_num, 'WUP2_7')
    copy2_7 = tf.concat([conv2, up2_7], 4)
    conv9_7 = Conv3d(copy2_7, 3, 1, 3*base_num, base_num, 'SAME', 'W9_7')
    conv10_7 = Conv3d(conv9_7, 3, 1, base_num, base_num/2, 'SAME', 'W10_7')
    conv11_7 = Conv3d(conv10_7, 3, 1, base_num/2, base_num/4, 'SAME', 'W11_7')
    temp_rsn7 = Conv3d(conv11_7, 3, 1, base_num/4, 1, 'SAME', 'W12_7')
    rsn7 = tf.reshape(temp_rsn7, [48, 56, 48], name='rsn7_result')
    # rsn8
    attention8 = GAT(attention, 'AW8', 'Q8', 'K8')
    conv7_8 = Conv3d(attention8, 3, 1, 4*base_num, 4*base_num, 'SAME', 'W7_8')
    up1_8 = Deconv3d(conv7_8, 24, 28, 24, 4*base_num, 4*base_num, 'WUP1_8')
    copy1_8 = tf.concat([conv4, up1_8], 4)
    conv8_8 = Conv3d(copy1_8, 3, 1, 6*base_num, 2*base_num, 'SAME', 'W8_8')
    up2_8 = Deconv3d(conv8_8, 48, 56, 48, 2*base_num, 2*base_num, 'WUP2_8')
    copy2_8 = tf.concat([conv2, up2_8], 4)
    conv9_8 = Conv3d(copy2_8, 3, 1, 3*base_num, base_num, 'SAME', 'W9_8')
    conv10_8 = Conv3d(conv9_8, 3, 1, base_num, base_num/2, 'SAME', 'W10_8')
    conv11_8 = Conv3d(conv10_8, 3, 1, base_num/2, base_num/4, 'SAME', 'W11_8')
    temp_rsn8 = Conv3d(conv11_8, 3, 1, base_num/4, 1, 'SAME', 'W12_8')
    rsn8 = tf.reshape(temp_rsn8, [48, 56, 48], name='rsn8_result')
    # rsn9
    attention9 = GAT(attention, 'AW9', 'Q9', 'K9')
    conv7_9 = Conv3d(attention9, 3, 1, 4*base_num, 4*base_num, 'SAME', 'W7_9')
    up1_9 = Deconv3d(conv7_9, 24, 28, 24, 4*base_num, 4*base_num, 'WUP1_9')
    copy1_9 = tf.concat([conv4, up1_9], 4)
    conv8_9 = Conv3d(copy1_9, 3, 1, 6*base_num, 2*base_num, 'SAME', 'W8_9')
    up2_9 = Deconv3d(conv8_9, 48, 56, 48, 2*base_num, 2*base_num, 'WUP2_9')
    copy2_9 = tf.concat([conv2, up2_9], 4)
    conv9_9 = Conv3d(copy2_9, 3, 1, 3*base_num, base_num, 'SAME', 'W9_9')
    conv10_9 = Conv3d(conv9_9, 3, 1, base_num, base_num/2, 'SAME', 'W10_9')
    conv11_9 = Conv3d(conv10_9, 3, 1, base_num/2, base_num/4, 'SAME', 'W11_9')
    temp_rsn9 = Conv3d(conv11_9, 3, 1, base_num/4, 1, 'SAME', 'W12_9')
    rsn9 = tf.reshape(temp_rsn9, [48, 56, 48], name='rsn9_result')

    return rsn0, rsn1, rsn2, rsn3, rsn4, rsn5, rsn6, rsn7, rsn8, rsn9


def negative_or(x, y):
    temp = tf.concat([tf.reshape(x, (48, 56, 48, 1)), tf.reshape(y, (48, 56, 48, 1))], axis=3)
    loss_top = tf.reduce_sum(tf.reduce_min(temp, axis=3, keepdims=False))
    loss_bottom = (tf.reduce_sum(x)+tf.reduce_sum(y))/2
    output = -loss_top/loss_bottom
    return output


# input
x = tf.placeholder(dtype=tf.float32, shape=[1, 48, 56, 48, 176], name='input_x')
y0 = tf.placeholder(dtype=tf.float32, shape=[48, 56, 48], name='rsn_0')
y1 = tf.placeholder(dtype=tf.float32, shape=[48, 56, 48], name='rsn_1')
y2 = tf.placeholder(dtype=tf.float32, shape=[48, 56, 48], name='rsn_2')
y3 = tf.placeholder(dtype=tf.float32, shape=[48, 56, 48], name='rsn_3')
y4 = tf.placeholder(dtype=tf.float32, shape=[48, 56, 48], name='rsn_4')
y5 = tf.placeholder(dtype=tf.float32, shape=[48, 56, 48], name='rsn_5')
y6 = tf.placeholder(dtype=tf.float32, shape=[48, 56, 48], name='rsn_6')
y7 = tf.placeholder(dtype=tf.float32, shape=[48, 56, 48], name='rsn_7')
y8 = tf.placeholder(dtype=tf.float32, shape=[48, 56, 48], name='rsn_8')
y9 = tf.placeholder(dtype=tf.float32, shape=[48, 56, 48], name='rsn_9')
train_num = 160
epoch_num = 200
batch_size = 1
lr = 0.00001

predict_0, predict_1, predict_2, predict_3, predict_4, predict_5, predict_6, predict_7, predict_8, predict_9 = Xinet(x)
loss0 = negative_or(predict_0, y0)
loss1 = negative_or(predict_1, y1)
loss2 = negative_or(predict_2, y2)
loss3 = negative_or(predict_3, y3)
loss4 = negative_or(predict_4, y4)
loss5 = negative_or(predict_5, y5)
loss6 = negative_or(predict_6, y6)
loss7 = negative_or(predict_7, y7)
loss8 = negative_or(predict_8, y8)
loss9 = negative_or(predict_9, y9)
loss = loss0+loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8+loss9
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

# save
saver = tf.train.Saver(max_to_keep=200)
# begin train
sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

x_r, y_s_r = load_data()
for epoch in tqdm(range(epoch_num)):
    # train
    path = '/media/D/alex/MIA_MH_GAGNN/MH_GAGNN_train/fold1/spatial/result_mat/epoch%s' % (epoch+1)
    os.mkdir(path)
    for i in range(train_num):
        epoch_x = x_r[i, :, :, :, :].reshape(1, 48, 56, 48, 176)
        epoch_y_0 = y_s_r[0+10*i, :, :, :].reshape(48, 56, 48)
        epoch_y_1 = y_s_r[1+10*i, :, :, :].reshape(48, 56, 48)
        epoch_y_2 = y_s_r[2+10*i, :, :, :].reshape(48, 56, 48)
        epoch_y_3 = y_s_r[3+10*i, :, :, :].reshape(48, 56, 48)
        epoch_y_4 = y_s_r[4+10*i, :, :, :].reshape(48, 56, 48)
        epoch_y_5 = y_s_r[5+10*i, :, :, :].reshape(48, 56, 48)
        epoch_y_6 = y_s_r[6+10*i, :, :, :].reshape(48, 56, 48)
        epoch_y_7 = y_s_r[7+10*i, :, :, :].reshape(48, 56, 48)
        epoch_y_8 = y_s_r[8+10*i, :, :, :].reshape(48, 56, 48)
        epoch_y_9 = y_s_r[9+10*i, :, :, :].reshape(48, 56, 48)
        sess.run(optimizer, feed_dict={x: epoch_x, y0: epoch_y_0, y1: epoch_y_1, y2: epoch_y_2, y3: epoch_y_3, y4: epoch_y_4, y5: epoch_y_5, y6: epoch_y_6, y7: epoch_y_7, y8: epoch_y_8, y9: epoch_y_9})
    saver.save(sess, r'/media/D/alex/MIA_MH_GAGNN/MH_GAGNN_train/fold1/spatial/Xinet.ckpt-' + str(epoch+1))

    # cal_loss
    epoch_loss = 0
    epoch_loss0, epoch_loss1, epoch_loss2, epoch_loss3, epoch_loss4, epoch_loss5, epoch_loss6, epoch_loss7, epoch_loss8, epoch_loss9 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    for i in range(train_num):
        epoch_x = x_r[i, :, :, :, :].reshape(1, 48, 56, 48, 176)
        epoch_y_0 = y_s_r[0+10*i, :, :, :].reshape(48, 56, 48)
        epoch_y_1 = y_s_r[1+10*i, :, :, :].reshape(48, 56, 48)
        epoch_y_2 = y_s_r[2+10*i, :, :, :].reshape(48, 56, 48)
        epoch_y_3 = y_s_r[3+10*i, :, :, :].reshape(48, 56, 48)
        epoch_y_4 = y_s_r[4+10*i, :, :, :].reshape(48, 56, 48)
        epoch_y_5 = y_s_r[5+10*i, :, :, :].reshape(48, 56, 48)
        epoch_y_6 = y_s_r[6+10*i, :, :, :].reshape(48, 56, 48)
        epoch_y_7 = y_s_r[7+10*i, :, :, :].reshape(48, 56, 48)
        epoch_y_8 = y_s_r[8+10*i, :, :, :].reshape(48, 56, 48)
        epoch_y_9 = y_s_r[9+10*i, :, :, :].reshape(48, 56, 48)
        loss_total, loss_rsn0, loss_rsn1, loss_rsn2, loss_rsn3, loss_rsn4, loss_rsn5, loss_rsn6, loss_rsn7, loss_rsn8, loss_rsn9, test0, test1, test2, test3, test4, test5, test6, test7, test8, test9 = sess.run([loss, loss0, loss1, loss2, loss3, loss4, loss5, loss6, loss7, loss8, loss9, predict_0, predict_1, predict_2, predict_3, predict_4, predict_5, predict_6, predict_7, predict_8, predict_9], feed_dict={x: epoch_x, y0: epoch_y_0, y1: epoch_y_1, y2: epoch_y_2, y3: epoch_y_3, y4: epoch_y_4, y5: epoch_y_5, y6: epoch_y_6, y7: epoch_y_7, y8: epoch_y_8, y9: epoch_y_9})
        epoch_loss += loss_total / train_num
        epoch_loss0 += loss_rsn0 / train_num
        epoch_loss1 += loss_rsn1 / train_num
        epoch_loss2 += loss_rsn2 / train_num
        epoch_loss3 += loss_rsn3 / train_num
        epoch_loss4 += loss_rsn4 / train_num
        epoch_loss5 += loss_rsn5 / train_num
        epoch_loss6 += loss_rsn6 / train_num
        epoch_loss7 += loss_rsn7 / train_num
        epoch_loss8 += loss_rsn8 / train_num
        epoch_loss9 += loss_rsn9 / train_num
        path = '/media/D/alex/MIA_MH_GAGNN/MH_GAGNN_train/fold1/spatial/result_mat/epoch%s/%s' % (epoch+1, i+1)
        os.mkdir(path)
        save0 = '/media/D/alex/MIA_MH_GAGNN/MH_GAGNN_train/fold1/spatial/result_mat/epoch%s/%s/rsn1.mat' % (epoch+1, i+1)
        scio.savemat(save0, {'space': test0})
        save1 = '/media/D/alex/MIA_MH_GAGNN/MH_GAGNN_train/fold1/spatial/result_mat/epoch%s/%s/rsn2.mat' % (epoch+1, i + 1)
        scio.savemat(save1, {'space': test1})
        save2 = '/media/D/alex/MIA_MH_GAGNN/MH_GAGNN_train/fold1/spatial/result_mat/epoch%s/%s/rsn3.mat' % (epoch+1, i + 1)
        scio.savemat(save2, {'space': test2})
        save3 = '/media/D/alex/MIA_MH_GAGNN/MH_GAGNN_train/fold1/spatial/result_mat/epoch%s/%s/rsn4.mat' % (epoch+1, i + 1)
        scio.savemat(save3, {'space': test3})
        save4 = '/media/D/alex/MIA_MH_GAGNN/MH_GAGNN_train/fold1/spatial/result_mat/epoch%s/%s/rsn5.mat' % (epoch+1, i + 1)
        scio.savemat(save4, {'space': test4})
        save5 = '/media/D/alex/MIA_MH_GAGNN/MH_GAGNN_train/fold1/spatial/result_mat/epoch%s/%s/rsn6.mat' % (epoch+1, i + 1)
        scio.savemat(save5, {'space': test5})
        save6 = '/media/D/alex/MIA_MH_GAGNN/MH_GAGNN_train/fold1/spatial/result_mat/epoch%s/%s/rsn7.mat' % (epoch+1, i + 1)
        scio.savemat(save6, {'space': test6})
        save7 = '/media/D/alex/MIA_MH_GAGNN/MH_GAGNN_train/fold1/spatial/result_mat/epoch%s/%s/rsn8.mat' % (epoch+1, i + 1)
        scio.savemat(save7, {'space': test7})
        save8 = '/media/D/alex/MIA_MH_GAGNN/MH_GAGNN_train/fold1/spatial/result_mat/epoch%s/%s/rsn9.mat' % (epoch+1, i + 1)
        scio.savemat(save8, {'space': test8})
        save9 = '/media/D/alex/MIA_MH_GAGNN/MH_GAGNN_train/fold1/spatial/result_mat/epoch%s/%s/rsn10.mat' % (epoch+1, i + 1)
        scio.savemat(save9, {'space': test9})
    print('Epoch: %03d/%03d train_loss: %.9f rsn1: %.9f rsn2: %.9f rsn3: %.9f rsn4: %.9f rsn5: %.9f rsn6: %.9f rsn7: %.9f rsn8: %.9f rsn9: %.9f rsn10: %.9f' % (epoch, epoch_num, epoch_loss, epoch_loss0, epoch_loss1, epoch_loss2, epoch_loss3, epoch_loss4, epoch_loss5, epoch_loss6, epoch_loss7, epoch_loss8, epoch_loss9))
    f = open('/media/D/alex/MIA_MH_GAGNN/MH_GAGNN_train/fold1/spatial/result', 'a')
    f.write('\n' + str(epoch_loss))
    f.write('\n' + str(epoch_loss0))
    f.write('\n' + str(epoch_loss1))
    f.write('\n' + str(epoch_loss2))
    f.write('\n' + str(epoch_loss3))
    f.write('\n' + str(epoch_loss4))
    f.write('\n' + str(epoch_loss5))
    f.write('\n' + str(epoch_loss6))
    f.write('\n' + str(epoch_loss7))
    f.write('\n' + str(epoch_loss8))
    f.write('\n' + str(epoch_loss9))
    f.close()


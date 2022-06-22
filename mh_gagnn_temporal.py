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
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def load_data():
    temp1 = []
    with open("/media/D/alex/MIA_MH_GAGNN/4DfMRI_input/make_fold/temporal1.txt", "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            index_id = int(line)
            train_path = '/media/D/alex/MIA_MH_GAGNN/4DfMRI_input/data_fMRI_48_56_48/%s/emotion/input_data.mat' % (index_id)
            train_data = loadmat(train_path)
            train_data = train_data['input_data']
            temp1.append(train_data)
    x = np.array(temp1)
    x = x.reshape(-1, 48, 56, 48, 176)  # x,y,z,t
    x = x.astype(np.float32)

    temp1 = []
    for rsn_id in range(10):
        with open("/media/D/alex/MIA_MH_GAGNN/4DfMRI_input/make_fold/temporal1.txt", "r") as f:
            for line in f.readlines():
                line = line.strip('\n')
                index_id = int(line)
                label_path = '/media/D/alex/MIA_MH_GAGNN/4DfMRI_input/dl_result/%s/emotion/rsn%s_temporal_label.mat' % (index_id, rsn_id+1)
                label_data = loadmat(label_path)
                label_data = label_data['time']
                temp1.append(label_data)
    y = np.array(temp1)
    y = y.reshape(-1, 176)  # x,y,z
    y_t = y.astype(np.float32)


    temp1 = []
    for rsn_id in range(10):
        with open("/media/D/alex/MIA_MH_GAGNN/4DfMRI_input/make_fold/temporal1.txt", "r") as f:
            for line in f.readlines():
                line = line.strip('\n')
                index_id = int(line)
                label_path = '/media/D/alex/MIA_MH_GAGNN/MH_GAGNN_test/fold1/emotion/%s/rsn%s.mat' % (index_id, rsn_id+1)
                label_data = loadmat(label_path)
                label_data = label_data['space']
                temp1.append(label_data)
    spatial_p = np.array(temp1)
    spatial_p = spatial_p.reshape(-1, 48, 56, 48)  # x,y,z
    spatial_p = spatial_p.astype(np.float32)

    return x, y_t, spatial_p


def Conv3d(x, k_size, s_size, c_in, c_out, pad_type, filter_name):
    filter_sets = tf.get_variable(filter_name, shape=[k_size, k_size, k_size, c_in, c_out], initializer=tf.keras.initializers.he_normal())
    c = tf.nn.conv3d(x, filter_sets, strides=[1, s_size, s_size, s_size, 1], padding=pad_type)
    bn = tf.keras.layers.BatchNormalization()
    c_temp = bn(c, training=True)
    output = tf.nn.relu(c_temp)
    return output


def fast_downsampling(x, w1, w2, w3):
    x1 = tf.transpose(x, [0, 2, 3, 4, 1])
    m1 = Conv3d(x1, 3, 1, 48, downsampling_size, 'SAME', w1)
    x2 = tf.transpose(m1, [0, 4, 2, 3, 1])
    m2 = Conv3d(x2, 3, 1, 56, downsampling_size, 'SAME', w2)
    x3 = tf.transpose(m2, [0, 1, 4, 3, 2])
    m3 = Conv3d(x3, 3, 1, 48, downsampling_size, 'SAME', w3)
    output = tf.reshape(tf.transpose(m3, [0, 1, 2, 4, 3]), [batch_size, downsampling_size*downsampling_size*downsampling_size, 176])
    return output


def multi_guided_attention(x, spatial_template):
    # value
    temp_value = fast_downsampling(x, 'w1_v', 'w2_v', 'w3_v')
    # query
    template0 = spatial_template[0:batch_size, :, :, :]
    template1 = spatial_template[batch_size:2*batch_size, :, :, :]
    template2 = spatial_template[2*batch_size:3*batch_size, :, :, :]
    template3 = spatial_template[3*batch_size:4*batch_size, :, :, :]
    template4 = spatial_template[4*batch_size:5*batch_size, :, :, :]
    template5 = spatial_template[5*batch_size:6*batch_size, :, :, :]
    template6 = spatial_template[6*batch_size:7*batch_size, :, :, :]
    template7 = spatial_template[7*batch_size:8*batch_size, :, :, :]
    template8 = spatial_template[8*batch_size:9*batch_size, :, :, :]
    template9 = spatial_template[9*batch_size:10*batch_size, :, :, :]
    temp_x = tf.reshape(x, [batch_size, 129024, 176])
    # rsn0
    temp_key0 = fast_downsampling(x, 'w1_k0', 'w2_k0', 'w3_k0')
    temp_query0 = tf.multiply(temp_x, tf.tile(tf.reshape(template0, [batch_size, 129024, 1]), [1, 1, 176]))
    query0 = fast_downsampling(tf.reshape(temp_query0, [batch_size, 48, 56, 48, 176]), 'w1_q0', 'w2_q0', 'w3_q0')
    # b, downsampling_size^3, 176
    temp_attention0 = tf.nn.softmax(tf.matmul(query0, tf.transpose(temp_key0, [0, 2, 1])), 2)
    result_time0 = tf.reshape(tf.reduce_mean(tf.transpose(tf.matmul(temp_attention0, temp_value), [0, 2, 1]), 2), [batch_size, 176], name='result0')
    # rsn1
    temp_key1 = fast_downsampling(x, 'w1_k1', 'w2_k1', 'w3_k1')
    temp_query1 = tf.multiply(temp_x, tf.tile(tf.reshape(template1, [batch_size, 129024, 1]), [1, 1, 176]))
    query1 = fast_downsampling(tf.reshape(temp_query1, [batch_size, 48, 56, 48, 176]), 'w1_q1', 'w2_q1', 'w3_q1')
    temp_attention1 = tf.nn.softmax(tf.matmul(query1, tf.transpose(temp_key1, [0, 2, 1])), 2)
    result_time1 = tf.reshape(tf.reduce_mean(tf.transpose(tf.matmul(temp_attention1, temp_value), [0, 2, 1]), 2), [batch_size, 176], name='result1')
    # rsn2
    temp_key2 = fast_downsampling(x, 'w1_k2', 'w2_k2', 'w3_k2')
    temp_query2 = tf.multiply(temp_x, tf.tile(tf.reshape(template2, [batch_size, 129024, 1]), [1, 1, 176]))
    query2 = fast_downsampling(tf.reshape(temp_query2, [batch_size, 48, 56, 48, 176]), 'w1_q2', 'w2_q2', 'w3_q2')
    temp_attention2 = tf.nn.softmax(tf.matmul(query2, tf.transpose(temp_key2, [0, 2, 1])), 2)
    result_time2 = tf.reshape(tf.reduce_mean(tf.transpose(tf.matmul(temp_attention2, temp_value), [0, 2, 1]), 2), [batch_size, 176], name='result2')
    # rsn3
    temp_key3 = fast_downsampling(x, 'w1_k3', 'w2_k3', 'w3_k3')
    temp_query3 = tf.multiply(temp_x, tf.tile(tf.reshape(template3, [batch_size, 129024, 1]), [1, 1, 176]))
    query3 = fast_downsampling(tf.reshape(temp_query3, [batch_size, 48, 56, 48, 176]), 'w1_q3', 'w2_q3', 'w3_q3')
    temp_attention3 = tf.nn.softmax(tf.matmul(query3, tf.transpose(temp_key3, [0, 2, 1])), 2)
    result_time3 = tf.reshape(tf.reduce_mean(tf.transpose(tf.matmul(temp_attention3, temp_value), [0, 2, 1]), 2), [batch_size, 176], name='result3')
    # rsn4
    temp_key4 = fast_downsampling(x, 'w1_k4', 'w2_k4', 'w3_k4')
    temp_query4 = tf.multiply(temp_x, tf.tile(tf.reshape(template4, [batch_size, 129024, 1]), [1, 1, 176]))
    query4 = fast_downsampling(tf.reshape(temp_query4, [batch_size, 48, 56, 48, 176]), 'w1_q4', 'w2_q4', 'w3_q4')
    temp_attention4 = tf.nn.softmax(tf.matmul(query4, tf.transpose(temp_key4, [0, 2, 1])), 2)
    result_time4 = tf.reshape(tf.reduce_mean(tf.transpose(tf.matmul(temp_attention4, temp_value), [0, 2, 1]), 2), [batch_size, 176], name='result4')
    # rsn5
    temp_key5 = fast_downsampling(x, 'w1_k5', 'w2_k5', 'w3_k5')
    temp_query5 = tf.multiply(temp_x, tf.tile(tf.reshape(template5, [batch_size, 129024, 1]), [1, 1, 176]))
    query5 = fast_downsampling(tf.reshape(temp_query5, [batch_size, 48, 56, 48, 176]), 'w1_q5', 'w2_q5', 'w3_q5')
    temp_attention5 = tf.nn.softmax(tf.matmul(query5, tf.transpose(temp_key5, [0, 2, 1])), 2)
    result_time5 = tf.reshape(tf.reduce_mean(tf.transpose(tf.matmul(temp_attention5, temp_value), [0, 2, 1]), 2), [batch_size, 176], name='result5')
    # rsn6
    temp_key6 = fast_downsampling(x, 'w1_k6', 'w2_k6', 'w3_k6')
    temp_query6 = tf.multiply(temp_x, tf.tile(tf.reshape(template6, [batch_size, 129024, 1]), [1, 1, 176]))
    query6 = fast_downsampling(tf.reshape(temp_query6, [batch_size, 48, 56, 48, 176]), 'w1_q6', 'w2_q6', 'w3_q6')
    temp_attention6 = tf.nn.softmax(tf.matmul(query6, tf.transpose(temp_key6, [0, 2, 1])), 2)
    result_time6 = tf.reshape(tf.reduce_mean(tf.transpose(tf.matmul(temp_attention6, temp_value), [0, 2, 1]), 2), [batch_size, 176], name='result6')
    # rsn7
    temp_key7 = fast_downsampling(x, 'w1_k7', 'w2_k7', 'w3_k7')
    temp_query7 = tf.multiply(temp_x, tf.tile(tf.reshape(template7, [batch_size, 129024, 1]), [1, 1, 176]))
    query7 = fast_downsampling(tf.reshape(temp_query7, [batch_size, 48, 56, 48, 176]), 'w1_q7', 'w2_q7', 'w3_q7')
    temp_attention7 = tf.nn.softmax(tf.matmul(query7, tf.transpose(temp_key7, [0, 2, 1])), 2)
    result_time7 = tf.reshape(tf.reduce_mean(tf.transpose(tf.matmul(temp_attention7, temp_value), [0, 2, 1]), 2), [batch_size, 176], name='result7')
    # rsn8
    temp_key8 = fast_downsampling(x, 'w1_k8', 'w2_k8', 'w3_k8')
    temp_query8 = tf.multiply(temp_x, tf.tile(tf.reshape(template8, [batch_size, 129024, 1]), [1, 1, 176]))
    query8 = fast_downsampling(tf.reshape(temp_query8, [batch_size, 48, 56, 48, 176]), 'w1_q8', 'w2_q8', 'w3_q8')
    temp_attention8 = tf.nn.softmax(tf.matmul(query8, tf.transpose(temp_key8, [0, 2, 1])), 2)
    result_time8 = tf.reshape(tf.reduce_mean(tf.transpose(tf.matmul(temp_attention8, temp_value), [0, 2, 1]), 2), [batch_size, 176], name='result8')
    # rsn9
    temp_key9 = fast_downsampling(x, 'w1_k9', 'w2_k9', 'w3_k9')
    temp_query9 = tf.multiply(temp_x, tf.tile(tf.reshape(template9, [batch_size, 129024, 1]), [1, 1, 176]))
    query9 = fast_downsampling(tf.reshape(temp_query9, [batch_size, 48, 56, 48, 176]), 'w1_q9', 'w2_q9', 'w3_q9')
    temp_attention9 = tf.nn.softmax(tf.matmul(query9, tf.transpose(temp_key9, [0, 2, 1])), 2)
    result_time9 = tf.reshape(tf.reduce_mean(tf.transpose(tf.matmul(temp_attention9, temp_value), [0, 2, 1]), 2), [batch_size, 176], name='result9')
    
    return result_time0, result_time1, result_time2, result_time3, result_time4, result_time5, result_time6, result_time7, result_time8, result_time9


def pearson_corr(y_t, predict_t):
    n = predict_t.shape[0].value
    n = tf.cast(n, dtype=tf.float32)
    sum_y = tf.reduce_sum(y_t)
    sum_p = tf.reduce_sum(predict_t)
    sum_y_sq = tf.reduce_sum(tf.square(y_t))
    sum_p_sq = tf.reduce_sum(tf.square(predict_t))
    sum_mul= tf.reduce_sum(tf.multiply(y_t, predict_t))
    unmerator = tf.subtract(tf.multiply(n, sum_mul), tf.multiply(sum_y, sum_p))
    denominator1 = tf.subtract(tf.multiply(n, sum_y_sq), tf.square(sum_y))
    denominator2 = tf.subtract(tf.multiply(n, sum_p_sq), tf.square(sum_p))
    denominator = tf.multiply(tf.sqrt(denominator1), tf.sqrt(denominator2))
    corr = tf.divide(unmerator, denominator)
    return tf.cond(tf.is_nan(corr), lambda: 0., lambda: corr)


# input
batch_size = 4
x = tf.placeholder(dtype=tf.float32, shape=[batch_size, 48, 56, 48, 176], name='input_x')
spatial = tf.placeholder(dtype=tf.float32, shape=[batch_size*10, 48, 56, 48], name='spatial_pattern')
yt0 = tf.placeholder(dtype=tf.float32, shape=[batch_size, 176], name='label_t0')
yt1 = tf.placeholder(dtype=tf.float32, shape=[batch_size, 176], name='label_t1')
yt2 = tf.placeholder(dtype=tf.float32, shape=[batch_size, 176], name='label_t2')
yt3 = tf.placeholder(dtype=tf.float32, shape=[batch_size, 176], name='label_t3')
yt4 = tf.placeholder(dtype=tf.float32, shape=[batch_size, 176], name='label_t4')
yt5 = tf.placeholder(dtype=tf.float32, shape=[batch_size, 176], name='label_t5')
yt6 = tf.placeholder(dtype=tf.float32, shape=[batch_size, 176], name='label_t6')
yt7 = tf.placeholder(dtype=tf.float32, shape=[batch_size, 176], name='label_t7')
yt8 = tf.placeholder(dtype=tf.float32, shape=[batch_size, 176], name='label_t8')
yt9 = tf.placeholder(dtype=tf.float32, shape=[batch_size, 176], name='label_t9')
train_num = 160
test_num=40
sub_num=200
epoch_num = 100
lr = 0.001
downsampling_size = 6

predict_0, predict_1, predict_2, predict_3, predict_4, predict_5, predict_6, predict_7, predict_8, predict_9 = multi_guided_attention(x, spatial)
loss_temp = [None]*batch_size
for i in range(batch_size):
    loss_temp[i] = -pearson_corr(tf.reshape(yt0[i,:],[176]), tf.reshape(predict_0[i,:],[176]))
loss_array = tf.stack(loss_temp, axis=0)
loss0 = tf.reduce_mean(loss_array)
loss_temp = [None]*batch_size
for i in range(batch_size):
    loss_temp[i] = -pearson_corr(tf.reshape(yt1[i,:],[176]), tf.reshape(predict_1[i,:],[176]))
loss_array = tf.stack(loss_temp, axis=0)
loss1 = tf.reduce_mean(loss_array)
loss_temp = [None]*batch_size
for i in range(batch_size):
    loss_temp[i] = -pearson_corr(tf.reshape(yt2[i,:],[176]), tf.reshape(predict_2[i,:],[176]))
loss_array = tf.stack(loss_temp, axis=0)
loss2 = tf.reduce_mean(loss_array)
loss_temp = [None]*batch_size
for i in range(batch_size):
    loss_temp[i] = -pearson_corr(tf.reshape(yt3[i,:],[176]), tf.reshape(predict_3[i,:],[176]))
loss_array = tf.stack(loss_temp, axis=0)
loss3 = tf.reduce_mean(loss_array)
loss_temp = [None]*batch_size
for i in range(batch_size):
    loss_temp[i] = -pearson_corr(tf.reshape(yt4[i,:],[176]), tf.reshape(predict_4[i,:],[176]))
loss_array = tf.stack(loss_temp, axis=0)
loss4 = tf.reduce_mean(loss_array)
loss_temp = [None]*batch_size
for i in range(batch_size):
    loss_temp[i] = -pearson_corr(tf.reshape(yt5[i,:],[176]), tf.reshape(predict_5[i,:],[176]))
loss_array = tf.stack(loss_temp, axis=0)
loss5 = tf.reduce_mean(loss_array)
loss_temp = [None]*batch_size
for i in range(batch_size):
    loss_temp[i] = -pearson_corr(tf.reshape(yt6[i,:],[176]), tf.reshape(predict_6[i,:],[176]))
loss_array = tf.stack(loss_temp, axis=0)
loss6 = tf.reduce_mean(loss_array)
loss_temp = [None]*batch_size
for i in range(batch_size):
    loss_temp[i] = -pearson_corr(tf.reshape(yt7[i,:],[176]), tf.reshape(predict_7[i,:],[176]))
loss_array = tf.stack(loss_temp, axis=0)
loss7 = tf.reduce_mean(loss_array)
loss_temp = [None]*batch_size
for i in range(batch_size):
    loss_temp[i] = -pearson_corr(tf.reshape(yt8[i,:],[176]), tf.reshape(predict_8[i,:],[176]))
loss_array = tf.stack(loss_temp, axis=0)
loss8 = tf.reduce_mean(loss_array)
loss_temp = [None]*batch_size
for i in range(batch_size):
    loss_temp[i] = -pearson_corr(tf.reshape(yt9[i,:],[176]), tf.reshape(predict_9[i,:],[176]))
loss_array = tf.stack(loss_temp, axis=0)
loss9 = tf.reduce_mean(loss_array)

loss_real = loss0+1.5*loss1+loss2+loss3+4*loss4+loss5+loss6+loss7+loss8+loss9
optimizer = tf.train.AdamOptimizer(lr).minimize(loss_real)

# save
saver = tf.train.Saver(max_to_keep=100)

# begin train
sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
x_r, y_t_r, spatial_r = load_data()
for epoch in tqdm(range(epoch_num)):
    path = '/media/D/alex/MIA_MH_GAGNN/MH_GAGNN_train/fold1/temporal/result_mat/epoch%s' % (epoch+1)
    os.mkdir(path)
    # train
    for i in range(int(train_num/batch_size)):
        epoch_x = x_r[i*batch_size:(i+1)*batch_size, :].reshape(batch_size, 48, 56, 48, 176)
        epoch_yt0 = y_t_r[i*batch_size:(i+1)*batch_size, :].reshape(batch_size, 176)
        epoch_yt1 = y_t_r[i*batch_size + sub_num*1:(i+1)*batch_size + sub_num*1, :].reshape(batch_size, 176)
        epoch_yt2 = y_t_r[i*batch_size + sub_num*2:(i+1)*batch_size + sub_num*2, :].reshape(batch_size, 176)
        epoch_yt3 = y_t_r[i*batch_size + sub_num*3:(i+1)*batch_size + sub_num*3, :].reshape(batch_size, 176)
        epoch_yt4 = y_t_r[i*batch_size + sub_num*4:(i+1)*batch_size + sub_num*4, :].reshape(batch_size, 176)
        epoch_yt5 = y_t_r[i*batch_size + sub_num*5:(i+1)*batch_size + sub_num*5, :].reshape(batch_size, 176)
        epoch_yt6 = y_t_r[i*batch_size + sub_num*6:(i+1)*batch_size + sub_num*6, :].reshape(batch_size, 176)
        epoch_yt7 = y_t_r[i*batch_size + sub_num*7:(i+1)*batch_size + sub_num*7, :].reshape(batch_size, 176)
        epoch_yt8 = y_t_r[i*batch_size + sub_num*8:(i+1)*batch_size + sub_num*8, :].reshape(batch_size, 176)
        epoch_yt9 = y_t_r[i*batch_size + sub_num*9:(i+1)*batch_size + sub_num*9, :].reshape(batch_size, 176)

        epoch_spatial0 = spatial_r[i * batch_size:(i + 1) * batch_size, :, :, :].reshape(batch_size, 48, 56, 48)
        epoch_spatial1 = spatial_r[i * batch_size + sub_num * 1:(i + 1) * batch_size + sub_num * 1, :, :, :].reshape(batch_size, 48, 56, 48)
        epoch_spatial2 = spatial_r[i * batch_size + sub_num * 2:(i + 1) * batch_size + sub_num * 2, :, :, :].reshape(batch_size, 48, 56, 48)
        epoch_spatial3 = spatial_r[i * batch_size + sub_num * 3:(i + 1) * batch_size + sub_num * 3, :, :, :].reshape(batch_size, 48, 56, 48)
        epoch_spatial4 = spatial_r[i * batch_size + sub_num * 4:(i + 1) * batch_size + sub_num * 4, :, :, :].reshape(batch_size, 48, 56, 48)
        epoch_spatial5 = spatial_r[i * batch_size + sub_num * 5:(i + 1) * batch_size + sub_num * 5, :, :, :].reshape(batch_size, 48, 56, 48)
        epoch_spatial6 = spatial_r[i * batch_size + sub_num * 6:(i + 1) * batch_size + sub_num * 6, :, :, :].reshape(batch_size, 48, 56, 48)
        epoch_spatial7 = spatial_r[i * batch_size + sub_num * 7:(i + 1) * batch_size + sub_num * 7, :, :, :].reshape(batch_size, 48, 56, 48)
        epoch_spatial8 = spatial_r[i * batch_size + sub_num * 8:(i + 1) * batch_size + sub_num * 8, :, :, :].reshape(batch_size, 48, 56, 48)
        epoch_spatial9 = spatial_r[i * batch_size + sub_num * 9:(i + 1) * batch_size + sub_num * 9, :, :, :].reshape(batch_size, 48, 56, 48)
        epoch_spatial = np.concatenate((epoch_spatial0, epoch_spatial1, epoch_spatial2, epoch_spatial3, epoch_spatial4, epoch_spatial5, epoch_spatial6, epoch_spatial7, epoch_spatial8, epoch_spatial9), axis=0)
        sess.run(optimizer, feed_dict={x: epoch_x, yt0: epoch_yt0, yt1: epoch_yt1, yt2: epoch_yt2, yt3: epoch_yt3, yt4: epoch_yt4, yt5: epoch_yt5, yt6: epoch_yt6, yt7: epoch_yt7, yt8: epoch_yt8, yt9: epoch_yt9, spatial: epoch_spatial})
    saver.save(sess, r'/media/D/alex/MIA_MH_GAGNN/MH_GAGNN_train/fold1/temporal/Xinet.ckpt-' + str(epoch+1))

    # cal_loss
    epoch_loss = 0
    epoch_loss0, epoch_loss1, epoch_loss2, epoch_loss3, epoch_loss4, epoch_loss5, epoch_loss6, epoch_loss7, epoch_loss8, epoch_loss9 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    for i in range(int(train_num/batch_size)):
        epoch_x = x_r[i*batch_size:(i+1)*batch_size, :].reshape(batch_size, 48, 56, 48, 176)
        epoch_yt0 = y_t_r[i*batch_size:(i+1)*batch_size, :].reshape(batch_size, 176)
        epoch_yt1 = y_t_r[i*batch_size + sub_num*1:(i+1)*batch_size + sub_num*1, :].reshape(batch_size, 176)
        epoch_yt2 = y_t_r[i*batch_size + sub_num*2:(i+1)*batch_size + sub_num*2, :].reshape(batch_size, 176)
        epoch_yt3 = y_t_r[i*batch_size + sub_num*3:(i+1)*batch_size + sub_num*3, :].reshape(batch_size, 176)
        epoch_yt4 = y_t_r[i*batch_size + sub_num*4:(i+1)*batch_size + sub_num*4, :].reshape(batch_size, 176)
        epoch_yt5 = y_t_r[i*batch_size + sub_num*5:(i+1)*batch_size + sub_num*5, :].reshape(batch_size, 176)
        epoch_yt6 = y_t_r[i*batch_size + sub_num*6:(i+1)*batch_size + sub_num*6, :].reshape(batch_size, 176)
        epoch_yt7 = y_t_r[i*batch_size + sub_num*7:(i+1)*batch_size + sub_num*7, :].reshape(batch_size, 176)
        epoch_yt8 = y_t_r[i*batch_size + sub_num*8:(i+1)*batch_size + sub_num*8, :].reshape(batch_size, 176)
        epoch_yt9 = y_t_r[i*batch_size + sub_num*9:(i+1)*batch_size + sub_num*9, :].reshape(batch_size, 176)

        epoch_spatial0 = spatial_r[i * batch_size:(i + 1) * batch_size, :, :, :].reshape(batch_size, 48, 56, 48)
        epoch_spatial1 = spatial_r[i * batch_size + sub_num * 1:(i + 1) * batch_size + sub_num * 1, :, :, :].reshape(batch_size, 48, 56, 48)
        epoch_spatial2 = spatial_r[i * batch_size + sub_num * 2:(i + 1) * batch_size + sub_num * 2, :, :, :].reshape(batch_size, 48, 56, 48)
        epoch_spatial3 = spatial_r[i * batch_size + sub_num * 3:(i + 1) * batch_size + sub_num * 3, :, :, :].reshape(batch_size, 48, 56, 48)
        epoch_spatial4 = spatial_r[i * batch_size + sub_num * 4:(i + 1) * batch_size + sub_num * 4, :, :, :].reshape(batch_size, 48, 56, 48)
        epoch_spatial5 = spatial_r[i * batch_size + sub_num * 5:(i + 1) * batch_size + sub_num * 5, :, :, :].reshape(batch_size, 48, 56, 48)
        epoch_spatial6 = spatial_r[i * batch_size + sub_num * 6:(i + 1) * batch_size + sub_num * 6, :, :, :].reshape(batch_size, 48, 56, 48)
        epoch_spatial7 = spatial_r[i * batch_size + sub_num * 7:(i + 1) * batch_size + sub_num * 7, :, :, :].reshape(batch_size, 48, 56, 48)
        epoch_spatial8 = spatial_r[i * batch_size + sub_num * 8:(i + 1) * batch_size + sub_num * 8, :, :, :].reshape(batch_size, 48, 56, 48)
        epoch_spatial9 = spatial_r[i * batch_size + sub_num * 9:(i + 1) * batch_size + sub_num * 9, :, :, :].reshape(batch_size, 48, 56, 48)
        epoch_spatial = np.concatenate((epoch_spatial0, epoch_spatial1, epoch_spatial2, epoch_spatial3, epoch_spatial4, epoch_spatial5, epoch_spatial6, epoch_spatial7, epoch_spatial8, epoch_spatial9), axis=0)
        loss_total, loss_rsn0, loss_rsn1, loss_rsn2, loss_rsn3, loss_rsn4, loss_rsn5, loss_rsn6, loss_rsn7, loss_rsn8, loss_rsn9, test0, test1, test2, test3, test4, test5, test6, test7, test8, test9 = sess.run(
            [loss_real, loss0, loss1, loss2, loss3, loss4, loss5, loss6, loss7, loss8, loss9, predict_0, predict_1,
             predict_2, predict_3, predict_4, predict_5, predict_6, predict_7, predict_8, predict_9],
            feed_dict={x: epoch_x, yt0: epoch_yt0, yt1: epoch_yt1, yt2: epoch_yt2, yt3: epoch_yt3, yt4: epoch_yt4, yt5: epoch_yt5, yt6: epoch_yt6, yt7: epoch_yt7, yt8: epoch_yt8, yt9: epoch_yt9, spatial: epoch_spatial})
        epoch_loss += loss_total / (train_num/batch_size)
        epoch_loss0 += loss_rsn0 / (train_num/batch_size)
        epoch_loss1 += loss_rsn1 / (train_num/batch_size)
        epoch_loss2 += loss_rsn2 / (train_num/batch_size)
        epoch_loss3 += loss_rsn3 / (train_num/batch_size)
        epoch_loss4 += loss_rsn4 / (train_num/batch_size)
        epoch_loss5 += loss_rsn5 / (train_num/batch_size)
        epoch_loss6 += loss_rsn6 / (train_num/batch_size)
        epoch_loss7 += loss_rsn7 / (train_num/batch_size)
        epoch_loss8 += loss_rsn8 / (train_num/batch_size)
        epoch_loss9 += loss_rsn9 / (train_num/batch_size)
    print(
        'Epoch: %03d/%03d train_loss: %.9f rsn1: %.9f rsn2: %.9f rsn3: %.9f rsn4: %.9f rsn5: %.9f rsn6: %.9f rsn7: %.9f rsn8: %.9f rsn9: %.9f rsn10: %.9f' % (
        epoch, epoch_num, epoch_loss, epoch_loss0, epoch_loss1, epoch_loss2, epoch_loss3, epoch_loss4, epoch_loss5,epoch_loss6, epoch_loss7, epoch_loss8, epoch_loss9))
    f = open('/media/D/alex/MIA_MH_GAGNN/MH_GAGNN_train/fold1/temporal/result', 'a')
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

    # val_loss
    epoch_loss = 0
    epoch_loss0, epoch_loss1, epoch_loss2, epoch_loss3, epoch_loss4, epoch_loss5, epoch_loss6, epoch_loss7, epoch_loss8, epoch_loss9 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    for i in range(int(test_num/batch_size)):
        epoch_x = x_r[160+i * batch_size:160+(i + 1) * batch_size, :, :, :, :].reshape(batch_size, 48, 56, 48, 176)
        epoch_yt0 = y_t_r[160+i * batch_size:160+(i + 1) * batch_size, :].reshape(batch_size, 176)
        epoch_yt1 = y_t_r[160+i * batch_size + sub_num * 1:160+(i + 1) * batch_size + sub_num * 1, :].reshape(
            batch_size, 176)
        epoch_yt2 = y_t_r[160+i * batch_size + sub_num * 2:160+(i + 1) * batch_size + sub_num * 2, :].reshape(
            batch_size, 176)
        epoch_yt3 = y_t_r[160+i * batch_size + sub_num * 3:160+(i + 1) * batch_size + sub_num * 3, :].reshape(
            batch_size, 176)
        epoch_yt4 = y_t_r[160+i * batch_size + sub_num * 4:160+(i + 1) * batch_size + sub_num * 4, :].reshape(
            batch_size, 176)
        epoch_yt5 = y_t_r[160+i * batch_size + sub_num * 5:160+(i + 1) * batch_size + sub_num * 5, :].reshape(
            batch_size, 176)
        epoch_yt6 = y_t_r[160+i * batch_size + sub_num * 6:160+(i + 1) * batch_size + sub_num * 6, :].reshape(
            batch_size, 176)
        epoch_yt7 = y_t_r[160+i * batch_size + sub_num * 7:160+(i + 1) * batch_size + sub_num * 7, :].reshape(
            batch_size, 176)
        epoch_yt8 = y_t_r[160+i * batch_size + sub_num * 8:160+(i + 1) * batch_size + sub_num * 8, :].reshape(
            batch_size, 176)
        epoch_yt9 = y_t_r[160+i * batch_size + sub_num * 9:160+(i + 1) * batch_size + sub_num * 9, :].reshape(
            batch_size, 176)

        epoch_spatial0 = spatial_r[160+i * batch_size:160+(i + 1) * batch_size, :, :, :].reshape(
            batch_size, 48, 56, 48)
        epoch_spatial1 = spatial_r[160+i * batch_size + sub_num * 1:160+(i + 1) * batch_size + sub_num * 1, :, :,
                         :].reshape(batch_size, 48, 56, 48)
        epoch_spatial2 = spatial_r[160+i * batch_size + sub_num * 2:160+(i + 1) * batch_size + sub_num * 2, :, :,
                         :].reshape(batch_size, 48, 56, 48)
        epoch_spatial3 = spatial_r[160+i * batch_size + sub_num * 3:160+(i + 1) * batch_size + sub_num * 3, :, :,
                         :].reshape(batch_size, 48, 56, 48)
        epoch_spatial4 = spatial_r[160+i * batch_size + sub_num * 4:160+(i + 1) * batch_size + sub_num * 4, :, :,
                         :].reshape(batch_size, 48, 56, 48)
        epoch_spatial5 = spatial_r[160+i * batch_size + sub_num * 5:160+(i + 1) * batch_size + sub_num * 5, :, :,
                         :].reshape(batch_size, 48, 56, 48)
        epoch_spatial6 = spatial_r[160+i * batch_size + sub_num * 6:160+(i + 1) * batch_size + sub_num * 6, :, :,
                         :].reshape(batch_size, 48, 56, 48)
        epoch_spatial7 = spatial_r[160+i * batch_size + sub_num * 7:160+(i + 1) * batch_size + sub_num * 7, :, :,
                         :].reshape(batch_size, 48, 56, 48)
        epoch_spatial8 = spatial_r[160+i * batch_size + sub_num * 8:160+(i + 1) * batch_size + sub_num * 8, :, :,
                         :].reshape(batch_size, 48, 56, 48)
        epoch_spatial9 = spatial_r[160+i * batch_size + sub_num * 9:160+(i + 1) * batch_size + sub_num * 9, :, :,
                         :].reshape(batch_size, 48, 56, 48)
        epoch_spatial = np.concatenate((epoch_spatial0, epoch_spatial1, epoch_spatial2, epoch_spatial3, epoch_spatial4,
                                        epoch_spatial5, epoch_spatial6, epoch_spatial7, epoch_spatial8, epoch_spatial9),
                                       axis=0)
        loss_total, loss_rsn0, loss_rsn1, loss_rsn2, loss_rsn3, loss_rsn4, loss_rsn5, loss_rsn6, loss_rsn7, loss_rsn8, loss_rsn9, test0, test1, test2, test3, test4, test5, test6, test7, test8, test9 = sess.run(
            [loss_real, loss0, loss1, loss2, loss3, loss4, loss5, loss6, loss7, loss8, loss9, predict_0, predict_1,
             predict_2, predict_3, predict_4, predict_5, predict_6, predict_7, predict_8, predict_9],
            feed_dict={x: epoch_x, yt0: epoch_yt0, yt1: epoch_yt1, yt2: epoch_yt2, yt3: epoch_yt3, yt4: epoch_yt4, yt5: epoch_yt5, yt6: epoch_yt6, yt7: epoch_yt7, yt8: epoch_yt8, yt9: epoch_yt9, spatial: epoch_spatial})
        epoch_loss += loss_total / (test_num/batch_size)
        epoch_loss0 += loss_rsn0 / (test_num/batch_size)
        epoch_loss1 += loss_rsn1 / (test_num/batch_size)
        epoch_loss2 += loss_rsn2 / (test_num/batch_size)
        epoch_loss3 += loss_rsn3 / (test_num/batch_size)
        epoch_loss4 += loss_rsn4 / (test_num/batch_size)
        epoch_loss5 += loss_rsn5 / (test_num/batch_size)
        epoch_loss6 += loss_rsn6 / (test_num/batch_size)
        epoch_loss7 += loss_rsn7 / (test_num/batch_size)
        epoch_loss8 += loss_rsn8 / (test_num/batch_size)
        epoch_loss9 += loss_rsn9 / (test_num/batch_size)
        for test_sub_id in range(batch_size):
            path = '/media/D/alex/MIA_MH_GAGNN/MH_GAGNN_train/fold1/temporal/result_mat/epoch%s/%s' % (epoch+1, batch_size*i + 1+test_sub_id)
            os.mkdir(path)
            temp_test0=test0[test_sub_id,:].reshape(1,176)
            save0 = path+'/rsn1.mat'
            scio.savemat(save0, {'time': temp_test0})
            temp_test1=test1[test_sub_id,:].reshape(1,176)
            save1 = path+'/rsn2.mat'
            scio.savemat(save1, {'time': temp_test1})
            temp_test2=test2[test_sub_id,:].reshape(1,176)
            save2 = path+'/rsn3.mat'
            scio.savemat(save2, {'time': temp_test2})
            temp_test3=test3[test_sub_id,:].reshape(1,176)
            save3 = path+'/rsn4.mat'
            scio.savemat(save3, {'time': temp_test3})
            temp_test4=test4[test_sub_id,:].reshape(1,176)
            save4 = path+'/rsn5.mat'
            scio.savemat(save4, {'time': temp_test4})
            temp_test5=test5[test_sub_id,:].reshape(1,176)
            save5 = path+'/rsn6.mat'
            scio.savemat(save5, {'time': temp_test5})
            temp_test6=test6[test_sub_id,:].reshape(1,176)
            save6 = path+'/rsn7.mat'
            scio.savemat(save6, {'time': temp_test6})
            temp_test7=test7[test_sub_id,:].reshape(1,176)
            save7 = path+'/rsn8.mat'
            scio.savemat(save7, {'time': temp_test7})
            temp_test8=test8[test_sub_id,:].reshape(1,176)
            save8 = path+'/rsn9.mat'
            scio.savemat(save8, {'time': temp_test8})
            temp_test9=test9[test_sub_id,:].reshape(1,176)
            save9 = path+'/rsn10.mat'
            scio.savemat(save9, {'time': temp_test9})
    print(
        'Epoch: %03d/%03d val_loss: %.9f rsn1: %.9f rsn2: %.9f rsn3: %.9f rsn4: %.9f rsn5: %.9f rsn6: %.9f rsn7: %.9f rsn8: %.9f rsn9: %.9f rsn10: %.9f' % (
        epoch, epoch_num, epoch_loss, epoch_loss0, epoch_loss1, epoch_loss2, epoch_loss3, epoch_loss4, epoch_loss5,epoch_loss6, epoch_loss7, epoch_loss8, epoch_loss9))
    f = open('/media/D/alex/MIA_MH_GAGNN/MH_GAGNN_train/fold1/temporal/result', 'a')
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


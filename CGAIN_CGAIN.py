import os, time, itertools, imageio, pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
tf.compat.v1.disable_v2_behavior

from sklearn.preprocessing import minmax_scale
from tqdm import tqdm

from CGAIN_utils import binary_sampler, normalization, sample_batch_index, uniform_sampler, renormalization, \
    rounding, rmse_loss, xavier_init

from CGAIN_data_loader import data_loader

# from sklearn.model_selection import KFold

def cgain(train_x, test_x, orig_train_x, orig_test_x, train_y, test_y, gain_parameters):

    '''Impute missing values in data_x conditional to y

    Args:
    - data_x: original data with missing values
    - gain_parameters: GAIN network parameters:
      - batch_size: Batch size
      - hint_rate: Hint rate
      - alpha: Hyperparameter
      - iterations: Iterations

    Returns:
    - imputed_data: imputed data
    '''
    tf.reset_default_graph()
    # System parameters
    batch_size = gain_parameters['batch_size']
    hint_rate = gain_parameters['hint_rate']
    alpha = gain_parameters['alpha']
    iterations = gain_parameters['iterations']

    # lr = 0.001

    # prepare cross validation
    # kfold = KFold(n_splits=5, shuffle=True, random_state=1)

    rmse_5fold = []

    # for train_idx, test_idx in kfold.split(miss_data_x_y):
    # train_x = miss_data_x_y[train_idx, :-1]
    # test_x = miss_data_x_y[test_idx, :-1]
    # ori_test_x = ori_data_x_y[test_idx, :-1]
    #
    # y = ori_data_x_y[:,-1]
    # y = y.reshape(y.shape[0], 1)
    train_y = train_y.reshape(train_y.shape[0],1)
    test_y = test_y.reshape(test_y.shape[0],1)
    no_classes = len(np.unique(np.append(train_y, test_y, axis=0)))
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(np.append(train_y, test_y, axis=0))
    train_y = enc.transform(train_y).toarray()
    test_y = enc.transform(test_y).toarray()

    train_y #= y[train_idx]
    test_y #= y[test_idx]

    # Define mask matrix
    train_m = 1-np.isnan(train_x)
    test_m = 1-np.isnan(test_x)

    # Other parameters
    no, dim = train_x.shape

    # Hidden state dimensions
    h_dim = int(dim)

    # Normalization
    norm_train, norm_parameters = normalization(train_x)
    norm_train_x = np.nan_to_num(norm_train, 0)
    norm_test, _ = normalization(test_x, norm_parameters)
    norm_test_x = np.nan_to_num(norm_test, 0)

    # Generator G(z)
    def generator(x, y, m, h_dim, dim, isTrain=True, reuse=False):
        tf.compat.v1.disable_eager_execution()
        with tf.compat.v1.variable_scope('generator', reuse=tf.compat.v1.AUTO_REUSE):
            w_init = tf.contrib.layers.xavier_initializer()

            cat1 = tf.compat.v1.concat([x, m, y], 1)

            dense1 = tf.compat.v1.layers.dense(cat1, dim, kernel_initializer=w_init)
            relu1 = tf.compat.v1.nn.tanh(dense1)

            dense2 = tf.compat.v1.layers.dense(relu1, h_dim, kernel_initializer=w_init)
            relu2 = tf.compat.v1.nn.tanh(dense2)

            g_logit = tf.compat.v1.layers.dense(relu2, dim, kernel_initializer=w_init)
            g_prob = tf.compat.v1.nn.sigmoid(g_logit)

            return g_prob

    # Discriminator D(x)
    def discriminator(x, y, h, h_dim, dim, isTrain=True, reuse=False):
        tf.compat.v1.disable_eager_execution()
        with tf.compat.v1.variable_scope('discriminator', reuse=reuse):#tf.compat.v1.AUTO_REUSE):
            w_init = tf.contrib.layers.xavier_initializer()

            cat1 = tf.compat.v1.concat([x, h, y], 1)

            dense1 = tf.compat.v1.layers.dense(cat1, dim, kernel_initializer=w_init)
            relu1 = tf.compat.v1.nn.tanh(dense1)

            dense2 = tf.compat.v1.layers.dense(relu1, h_dim, kernel_initializer=w_init)
            relu2 = tf.compat.v1.nn.tanh(dense2)

            d_logit = tf.compat.v1.layers.dense(relu2, dim, kernel_initializer=w_init)
            # d_logit = tf.layers.dense(relu2, 1, kernel_initializer=w_init)
            d_prob = tf.compat.v1.nn.sigmoid(d_logit)

            return d_prob, d_logit

        # nb_classes = len(trainy[0])
    # nb_classes = 2              # temporarily for diabetes dataset
    # dim = trainX.shape[1]
    # no = trainX.shape[0] + testX.shape[0]
    # h_dim = int(dim)


    # label preprocess
    onehot = np.eye(no_classes)

    temp_z_ = np.random.normal(0, 1, (10, 50))
    fixed_z_ = temp_z_
    fixed_y_ = np.zeros((10, 1))

    for i in range(1):
        fixed_z_ = np.concatenate([fixed_z_, temp_z_], 0)
        temp = np.ones((10,1)) + i
        fixed_y_ = np.concatenate([fixed_y_, temp], 0)

    fixed_y_ = onehot[fixed_y_.astype(np.int32)].squeeze()

    # variables : input
    tf.compat.v1.disable_eager_execution()
    x = tf.compat.v1.placeholder(tf.float32, shape=(None, dim))
    m = tf.compat.v1.placeholder(tf.float32, shape=(None, dim))
    h = tf.compat.v1.placeholder(tf.float32, shape=(None, dim))
    y = tf.compat.v1.placeholder(tf.float32, shape=(None, len(train_y[0])))
    z = tf.compat.v1.placeholder(tf.float32, shape=(None, 50))
    isTrain = tf.compat.v1.placeholder(dtype=tf.bool)

    # networks : generator
    G_sample = generator(x, y, m, h_dim, dim, isTrain)
    hat_x = x * m + G_sample * (1  - m)

    # networks : discriminator
    D_real, D_real_logits = discriminator(x, y, h, h_dim, dim, isTrain)
    D_fake, D_fake_logits = discriminator(hat_x, y, h, h_dim, dim, isTrain, reuse=True)

    # loss for each network
    D_loss_real = tf.compat.v1.reduce_mean(tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.compat.v1.ones([batch_size, dim])))
    D_loss_fake = tf.compat.v1.reduce_mean(tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.compat.v1.zeros([batch_size, dim])))
    D_loss_temp = -tf.compat.v1.reduce_mean(m * tf.compat.v1.math.log(D_fake + 1e-8) + (1 - m) * tf.compat.v1.math.log(1. - D_fake + 1e-8))

    D_loss = D_loss_real + D_loss_fake + D_loss_temp

    G_loss = tf.compat.v1.reduce_mean(tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones([batch_size, dim])))
    G_loss_temp = -tf.compat.v1.reduce_mean((1 - m) * tf.compat.v1.math.log(D_fake + 1e-8))
    MSE_loss = tf.compat.v1.reduce_mean((m * x - m * G_sample)**2) / tf.compat.v1.reduce_mean(m)

    G_loss = G_loss + G_loss_temp + alpha * MSE_loss

    # trainable variables for each network
    T_vars = tf.compat.v1.trainable_variables()
    D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
    G_vars = [var for var in T_vars if var.name.startswith('generator')]

    # optimizer for each network
    with tf.compat.v1.control_dependencies(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):
        D_optim = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001, beta1=0.5).minimize(D_loss, var_list=D_vars)
        G_optim = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001, beta1=0.5).minimize(G_loss, var_list=G_vars)

    # open session and initialize all variables
    sess = tf.compat.v1.InteractiveSession()
    tf.compat.v1.global_variables_initializer().run()

    train_hist = {}
    train_hist['D_losses'] = []
    train_hist['G_losses'] = []
    train_hist['per_epoch_ptimes'] = []
    train_hist['total_ptime'] = []

    # training-loop
    np.random.seed(int(time.time()))
    # print('training start!')
    start_time = time.time()

    for it in tqdm(range(iterations)):
        G_losses = []
        D_losses = []
        epoch_start_time = time.time()
        # for iter in range(len(train_set) // batch_size):

        batch_idx = sample_batch_index(train_x.shape[0], batch_size=batch_size)

        # update discriminator
        x_mb = norm_train_x[batch_idx, :]
        m_mb = train_m[batch_idx, :]
        y_mb = train_y[batch_idx, :]

        # random noise vector
        z_mb = uniform_sampler(0, 0.01, batch_size, dim)
        # hint vector
        h_mb_temp = binary_sampler(hint_rate, batch_size, dim)
        h_mb = m_mb * h_mb_temp

        # combine random noise with the observed x values
        x_mb = m_mb * x_mb + (1 - m_mb) * z_mb

        loss_d_, *_ = sess.run([D_loss, D_optim], {x: x_mb, y: y_mb, h: h_mb, m: m_mb, isTrain: True})
        D_losses.append(loss_d_)

        # update generator
        # z_ = np.random.normal(0, 1, (batch_size, 100))
        y_ = np.random.randint(0, 1, (batch_size, 1)) # mind the (number of classes-1)
        y_ = onehot[y_.astype(np.int32)].squeeze()
        loss_g_, *_ = sess.run([G_loss, G_optim, MSE_loss], {x: x_mb, m:m_mb, y:y_, h:h_mb, isTrain: True})
        G_losses.append(loss_g_)

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time
        # print('[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f' % ((it + 1), iterations, per_epoch_ptime, np.mean(D_losses), np.mean(G_losses)))
        # fixed_p = root + 'Fixed_results/' + model + str(it + 1) + '.png'
        # show_result((epoch + 1), save=True, path=fixed_p)
        train_hist['D_losses'].append(np.mean(D_losses))
        train_hist['G_losses'].append(np.mean(G_losses))
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

    end_time = time.time()
    total_ptime = end_time - start_time
    train_hist['total_ptime'].append(total_ptime)

    # print('Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f' % (np.mean(train_hist['per_epoch_ptimes']), iterations, total_ptime))
    # print("Training finish!... save training results")
    # with open(root + model + 'train_hist.pkl', 'wb') as f:
    #     pickle.dump(train_hist, f)

    # show_train_hist(train_hist, save=True, path=root + model + 'train_hist.png')

    # return imputed data
    z_mb = uniform_sampler(0, 0.01, test_x.shape[0], dim) # 9910 for news, 1150 for spam
    m_mb = test_m
    x_mb = norm_test_x
    x_mb = m_mb * x_mb + (1 - m_mb) * z_mb
    y_mb = test_y


    imputed_data = sess.run([G_sample], {x: x_mb, m:m_mb, y:y_mb, isTrain: False})
    # imputed_data = data_m * norm_data_x + (1 - data_m) * imputed_data
    imputed_data = m_mb * norm_test_x + (1 - m_mb) * imputed_data

    _, no, dim = imputed_data.shape

    # renormalization
    imputed_data = renormalization(imputed_data.reshape(no, dim), norm_parameters)

    # rounding
    imputed_data = rounding(imputed_data.reshape(no, dim), test_x)
    # np.savetxt('diabetes_imputed_MyCGAIN_20.csv', imputed_data)

    # Report the RMSE performance
    rmse = rmse_loss (orig_test_x, imputed_data, test_m)

    sess.close()

    return rmse, imputed_data, orig_test_x, test_m, test_y

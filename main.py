import tensorflow as tf
from CapsNet import *
import cv2
import numpy as np
from data import *

batch_size = 32
is_shift_ag = True
irun = 0
num_show = 5
lr = 0.001
steps = 100000
save_frequence = 1000
write_frequence = 500
decay_frequence = 5000
is_show_multi_rec = True
is_show_sample = True
key = -1
min_lr = 5e-6

train_iter = multi_train_iter(iters=steps,batch_size=batch_size,train = True)
test_iter =  multi_train_iter(iters=steps, batch_size=batch_size,train = False)
multi_iter = multi_train_iter(iters=steps,batch_size=num_show, train= False)

net = CapsNet(is_multi_mnist=is_multi_mnist)
tf.summary.scalar('error_rate_on_test_set', (1.0 - net.accuracy) * 100.0)
tf.summary.scalar('loss_reconstruction_on_test_set', net.loss_rec)
merged = tf.summary.merge_all()
init = tf.initialize_all_variables()

sess = tf.Session()
writer = tf.summary.FileWriter("./sum",sess.graph)
saver = tf.train.Saver()

sess.run(init)

for X,Y in train_iter:
    X_TEST, Y_TEST = test_iter.next()

    LS, LS_REC, ACC, _ = sess.run([net.loss, net.loss_rec, net.accuracy, net.train], feed_dict={net.x: X, net.y: Y, net.lr: lr})
    ACC_TEST, result = sess.run([net.accuracy,merged], feed_dict={net.x: X_TEST, net.y: Y_TEST})

    writer.add_summary(result, irun)

    print irun, LS, LS_REC, ACC, ACC_TEST
         
    if (irun + 1) % write_frequence == 0:
        X_MULTI,Y_MULTI = multi_iter.next()
        X_REC1,X_REC2 = sess.run(net.x_recs, feed_dict={net.x: X_MULTI, net.y: Y_MULTI})
 
        # turn the composed image to be 3 channel gray image
        images_org = np.stack([X_MULTI[:num_show,:,:,0]]*3,axis=-1)
        black = np.zeros([num_show, 36, 36, 1])
        images_recs = np.concatenate([black, X_REC1, X_REC2], axis=-1)
        images_rec1 = np.concatenate([black, black, X_REC2], axis=-1)
        images_rec2 = np.concatenate([black, X_REC1, black], axis=-1)
        image_show = np.concatenate([images_org, images_recs, images_rec1, images_rec2], axis=2)
        image_show = cv2.resize(np.concatenate(image_show, axis=0), dsize=(0, 0), fx=3, fy=3)
        np.save('MultiReconstruction_%d.png' % irun, image_show)
        cv2.imwrite('MultiRReconstruction_%d.png' % irun, image_show * 255.0)

    if (irun+1) % save_frequence == 0:
        saver.save(sess, './cpt/train_model', global_step=irun)

    if (irun+1) % decay_frequence == 0 and lr > min_lr:
        lr *= 0.5

    irun += 1

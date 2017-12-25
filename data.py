from tensorflow.examples.tutorials.mnist import input_data
import cv2
import numpy as np
import os
        yield augmentation(images,max_offset), np.stack([batch[1]]*3, axis=-1)

def one_hot(y, num_classes=10):
    y_hot = np.zeros((num_classes,))
    y_hot[y] = 1.0

    return y_hot

def multi_train_iter(iters=1000,batch_size=32,is_shift_ag=True, max_offset=4):

    train = []
    for subdir, dirs, files in os.walk('./masked'):
        for f in files:
            if ord(f[0]) > ord('J'):
                continue
            #if ord(f[0]) > ord('Z'):
            #    label = ord(f[0]) - 71
            else:
                label = ord(f[0]) - 65
            if '.png' not in f:
                continue
            label = one_hot(label)
            x = cv2.imread('./masked/' + f)

            gray_image = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            inv_image = abs(255 - gray_image)
            inv_image = inv_image / 255. 
            train.append((inv_image,label))

    for i in range(iters):
        count = 0
        images = np.zeros([batch_size, 36, 36, 3]) 
        y0 = []
        y1 = []
        y2 = []
        while count < batch_size:
            _image1, _y1 = train[np.random.randint(0, len(train))]
            _image2, _y2 = train[np.random.randint(0, len(train))]

            if np.array_equal(_y1, _y2):
                continue
            _image1 = augmentation(_image1, max_offset)
            _image2 = augmentation(_image2, max_offset)


            _images = np.maximum(_image1,_image2)
            _images = np.concatenate([_images,_image1,_image2], axis=-1)

            images[count] = _images 
    
            _y0 = np.logical_or(_y1,_y2).astype(np.float32)
             
            y0.append(_y0)
            y1.append(_y1)
            y2.append(_y2)
            
            count += 1

        yield images, np.stack([y0,y1,y2], axis=-1)


'''
m_train_iter = mnist_train_iter()
m_test_iter = mnist_test_iter()
mm_train_iter = multimnist_train_iter()
mm_test_iter = multimnist_test_iter()
for i in range(10):
    img_mtrain = m_train_iter.next()[0][0]
    img_mtest = m_test_iter.next()[0][0]
    img_mmtrain = mm_train_iter.next()[0][0]
    img_mmtest = mm_test_iter.next()[0][0]
    cv2.imshow('mnist_train',cv2.resize(img_mtrain,dsize=(128,128)))
    cv2.imshow('mnist_test', cv2.resize(img_mtest,dsize=(128,128)))
    cv2.imshow('multimnist_train', cv2.resize(img_mmtrain,dsize=(128,128)))
    cv2.imshow('multimnist_test', cv2.resize(img_mmtest,dsize=(128,128)))
    cv2.waitKey(-1)
'''

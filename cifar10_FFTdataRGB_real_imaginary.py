import  os
import  tensorflow as tf
import  argparse
import  numpy as np
import  matplotlib.pyplot as plt
import  cv2


print('loading data...')
(x_train,y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# all data: uint8
# N-H-W-C

x_train = x_train / 255.
x_test = x_test / 255.
mean = np.mean(x_train, axis=(0, 1, 2, 3))
std = np.std(x_train, axis=(0, 1, 2, 3))
print('mean:', mean, 'std:', std)
# x_train = (x_train - mean) / (std + 1e-7)
# X_test = (x_test - mean) / (std + 1e-7)

# left eye and right eye: around 80% overlap
# using corp_and_resize method
# tf.tile: 堆叠行
# define a function to generate and save all data

nTrain = x_train.shape[0]
nTest  = x_test.shape[0]
imHeight = x_train.shape[1]
imWidth  = x_train.shape[2]
imDepth  = x_train.shape[3]

# iTrain = 1
x_train_FFT_real = np.zeros_like(x_train, dtype=np.float)
x_test_FFT_real  = np.zeros_like(x_test, dtype=np.float)
x_train_FFT_imaginary = np.zeros_like(x_train, dtype=np.float)
x_test_FFT_imaginary  = np.zeros_like(x_test, dtype=np.float)

for iTrain in range(nTrain): # nTrain
    img = x_train[iTrain, :, :, :]
    # img = img / 255.0
    # print(img)
    # print(img.dtype)
    # img = 0.2126 * img[:,:,0] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,2] 
    log_shift2centerRGB_real = np.zeros_like(img, dtype=np.float)
    log_shift2centerRGB_imaginary = np.zeros_like(img, dtype=np.float)
    for iChannel in range(imDepth):
        fft2 = np.fft.fft2(img[:, :, iChannel])
        shift2center = fft2
        # shift2center = np.fft.fftshift(fft2)
        # log_fft2 = np.log(1 + np.abs(fft2))
        log_shift2centerRGB_real[:, :, iChannel] = shift2center.real
        # print(log_shift2centerRGB_real)
        log_shift2centerRGB_imaginary[:, :, iChannel] = shift2center.imag

        # print(type(log_shift2centerRGB))
    x_train_FFT_real[iTrain, :, :, :] = log_shift2centerRGB_real
    # print(log_shift2centerRGB_real)
    # print(x_train_FFT_real[iTrain, :, :, :])
    x_train_FFT_imaginary[iTrain, :, :, :] = log_shift2centerRGB_imaginary
    print('Current training sample ID:', iTrain, 'Overall:', nTrain)

for iTest in range(nTest):
    img = x_test[iTest, :, :, :]
    # img = img / 255.0
    # img = 0.2126 * img[:,:,0] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,2] 
    log_shift2centerRGB_real = np.zeros_like(img, dtype=np.float)
    log_shift2centerRGB_imaginary = np.zeros_like(img, dtype=np.float)
    for iChannel in range(imDepth):
        fft2 = np.fft.fft2(img[:, :, iChannel])
        shift2center = fft2
        # shift2center = np.fft.fftshift(fft2)
        # log_fft2 = np.log(1 + np.abs(fft2))
        log_shift2centerRGB_real[:, :, iChannel] = shift2center.real
        log_shift2centerRGB_imaginary[:, :, iChannel] = shift2center.imag
    x_test_FFT_real[iTest, :, :, :] = log_shift2centerRGB_real
    x_test_FFT_imaginary[iTest, :, :, :] = log_shift2centerRGB_imaginary
    print('Current test sample ID:', iTest, 'Overall:', nTest)


mean = np.mean(x_train_FFT_real, axis=(0, 1, 2, 3))
std = np.std(x_train_FFT_real, axis=(0, 1, 2, 3))
print('mean_fftr:', mean, 'std_fftr:', std)

mean = np.mean(x_test_FFT_real, axis=(0, 1, 2, 3))
std = np.std(x_test_FFT_real, axis=(0, 1, 2, 3))
print('mean_fftrtest:', mean, 'std_fftrtest:', std)

mean = np.mean(x_train_FFT_imaginary, axis=(0, 1, 2, 3))
std = np.std(x_train_FFT_imaginary, axis=(0, 1, 2, 3))
print('mean_ffti:', mean, 'std_ffti:', std)

mean = np.mean(x_test_FFT_imaginary, axis=(0, 1, 2, 3))
std = np.std(x_test_FFT_imaginary, axis=(0, 1, 2, 3))
print('mean_fftitest:', mean, 'std_fftitest:', std)

# x_train_FFT_real = np.around(x_train_FFT_real)
# x_train_FFT_imaginary = np.around(x_train_FFT_imaginary)
# x_test_FFT_real = np.around(x_test_FFT_real)
# x_test_FFT_imaginary = np.around(x_test_FFT_imaginary)

# print(x_train_FFT_real[1, :, :, 0], x_train_FFT_imaginary[1, :, :, 0])
# img = x_train_FFT_real[1, :, :, 0] # N-H-W-C
# plt.subplot(231),plt.imshow(img),plt.title('x_train_real1')

# img_row = img.reshape(1, -1)
# img_row_log = np.log(img_row)
# sorted_img = np.sort(img_row)
# sorted_img_log = np.sort(img_row_log)
# print(sorted_img)
# print(sorted_img_log)

# img2 = x_train_FFT_real[1, :, :, 1] # N-H-W-C
# plt.subplot(232),plt.imshow(img2),plt.title('x_train_real2')
#
# img3 = x_train_FFT_real[1, :, :, 2] # N-H-W-C
# plt.subplot(233),plt.imshow(img3),plt.title('x_train_real3')
#
# img4 = x_train_FFT_imaginary[1, :, :, 0] # N-H-W-C
# plt.subplot(234),plt.imshow(img4),plt.title('x_train_imaginary1')
#
# img5 = x_train_FFT_imaginary[1, :, :, 1] # N-H-W-C
# plt.subplot(235),plt.imshow(img5),plt.title('x_train_imaginary2')
#
# img6 = x_train_FFT_imaginary[1, :, :, 2] # N-H-W-C
# plt.subplot(236),plt.imshow(img6),plt.title('x_train_imaginary3')
# plt.show()
# img = 0.2126 * img[:,:,0] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,2]
# plt.subplot(232),plt.imshow(img,'gray'),plt.title('original')
 
# fft2 = np.fft.fft2(img)
# plt.subplot(233),plt.imshow(np.abs(fft2),'gray'),plt.title('fft2')

# shift2center = np.fft.fftshift(fft2)
# plt.subplot(234),plt.imshow(np.abs(shift2center),'gray'),plt.title('shift2center')

# log_fft2 = np.log(1 + np.abs(fft2))
# plt.subplot(235),plt.imshow(log_fft2,'gray'),plt.title('log_fft2')

# log_shift2center = np.log(1 + np.abs(shift2center))
# plt.subplot(236),plt.imshow(log_shift2center,'gray'),plt.title('log_shift2center')
# plt.show()


np.save("x_train_FFT_real_noshift.npy", x_train_FFT_real)
np.save("x_test_FFT_real_noshift.npy", x_test_FFT_real)
np.save("x_train_FFT_imaginary_noshift.npy", x_train_FFT_imaginary)
np.save("x_test_FFT_imaginary_noshift.npy", x_test_FFT_imaginary)

# np.save("x_train_righteyes.npy", x_train_righteyes)
# np.save("x_test_righteyes.npy", x_test_righteyes)

# print(onebox.shape, boxes.shape, output.shape)

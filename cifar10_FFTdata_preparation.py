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


np.save("x_train_FFT_real_noshift.npy", x_train_FFT_real)
np.save("x_test_FFT_real_noshift.npy", x_test_FFT_real)
np.save("x_train_FFT_imaginary_noshift.npy", x_train_FFT_imaginary)
np.save("x_test_FFT_imaginary_noshift.npy", x_test_FFT_imaginary)

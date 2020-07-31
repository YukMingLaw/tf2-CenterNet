import caffe
import numpy as np
import argparse
import matplotlib.pyplot as plt
import time
# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--weights', type=str, required=True)
parser.add_argument('--iter', type=int, required=True)
args = parser.parse_args()

caffe.set_mode_cpu()

net = caffe.Net(args.model,args.weights,caffe.TEST)

transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0))
im=caffe.io.load_image('./sense.jpg')
net.blobs['data'].data[...] = transformer.preprocess('data', im)

for i in range(0, args.iter):
    for i in range(10):
        start = time.time()
        net.forward()
        end = time.time()
        print('Running time:', (end - start))

    image = net.blobs['data'].data
    predicted = net.blobs['conv2d_2'].data

    plt.figure()
    image = np.transpose(image[0],(1,2,0))
    plt.imshow(image)
    plt.figure()
    plt.imshow(predicted[0][0],vmin=0, vmax=1)
    plt.show()

print ('Success!')
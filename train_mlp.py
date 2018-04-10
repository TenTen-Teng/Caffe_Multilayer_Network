
import os
import caffe
import matplotlib.pyplot as plt
import numpy as np
import numpy
import time

caffe.set_mode_gpu()

solver = caffe.SGDSolver('mlp_solver.prototxt')

niter =20000
test_interval = 100
train_loss = np.zeros(niter)
test_acc = np.zeros(int(np.ceil(niter / test_interval)))

start = time.time()
# the main solver loop
acc_list = []
for it in range(niter):
    solver.step(1)  # SGD by Caffe

    # store the train loss
    train_loss[it] = solver.net.blobs['loss'].data
    solver.test_nets[0].forward(start='input1')

    if it % test_interval == 0:
        acc=solver.test_nets[0].blobs['accuracy'].data
        acc_list.append(acc)
        print 'Iteration', it, 'testing...','accuracy:',acc
        test_acc[it // test_interval] = acc


end = time.time()
print "time: ", end-start

average_acc = np.array(acc_list)
print "average accuracy: ", np.mean(average_acc)






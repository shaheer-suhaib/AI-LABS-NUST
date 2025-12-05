import numpy as np
lr = 0.1
epochs = 1
W = np.zeros(3)
bias = 0
threshold =0.5

data = np.array([
    [1,0,0,1],
    [1,0,1,1],
    [1,1,1,1],
    [0,1,1,0]
]
)

train = data[:,:3]
test = data[:,-1]
features = 3


for ep_number in range(epochs+1):
    #
    print("epoch no"+str(ep_number))
    error_count = 0
    for sample in data:
        # forward pass
        output = np.dot(sample[0:3],W) + bias          # fixed slice 0:3
        if output>=threshold:
            output = 1
        else:
            output =0
        # error
        erroor =sample[3] - output
        if erroor !=0:                                 # fixed condition
            # updating
            for w_index in range(3):
                W[w_index] = W[w_index] + lr*erroor*sample[w_index]
            bias = bias+ lr*erroor
            error_count+=1

    if error_count == 0 :                               # moved outside inner loop
        print("all corrected classified")
        break

print("weights are")
print(W)
print("bias is ")
print(bias)

import pickle
import numpy as np
import h5py
import tensorflow as tf
import cvnn.layers as complex_layers
import matplotlib.pyplot as plt
from cvnn.initializers import ComplexGlorotUniform


# Load the dataset ...
#  You will need to separately download or generate this file
with open('/home/tahawaru/Documents/Reporting/Validation/RadioML2016/Dataset/RML2016.10a_dict.pkl', 'rb') as f:
    Xd = pickle.load(f, encoding='latin')

ls = list(Xd.keys())
xyz = np.array([Xd.get(ls[i]) for i in range(len(ls))])

# Identifying the different modulations
modulations = []
snrs = []
mod_index_shift = [0]  # indices where the modulation type shifts
snr_index_shift = [0]  # indices where the snr value shifts
# init_mod = ls[0][0]
# init_snr = ls[0][1]
# Extracting the 11 modulations at xDB where x = snr_dB
snr_dB = 10
x_data = []
x_label = []
for k, l in enumerate(ls):
    if l[0] not in modulations:
        modulations.append(l[0])
        mod_index_shift.append(k)
    if l[1] not in snrs:
        snrs.append(l[1])
        snr_index_shift.append(k)
    if l[1] == snr_dB:
        x_data.append(np.complex64(tf.complex(xyz[k, :xyz.shape[1], 0],  xyz[k, :xyz.shape[1], 1])))
        for n in range(xyz.shape[1]):
            x_label.append(modulations.index(l[0]))

x_data = np.reshape(x_data, (len(modulations) * xyz.shape[1], 16, 8, 1))
x_label = np.reshape(x_label, (len(modulations) * xyz.shape[1], 1))

# sanity check of the labels
# diffar = np.unique(x_label)

# Shuffle the index array containing the labels at one specific S(I)NR value
def shuffle_in_unisson_permute(x_ar1, x_ar2, num_permute):
    assert x_ar1.shape[0] == x_ar2.shape[0]
    n_elt = x_ar1.shape[0]
    indices = np.zeros(n_elt)
    for j in range(num_permute):
        indices = np.random.permutation(n_elt)
    return x_ar1[indices], x_ar2[indices]


# construct the dataset from the shuffled indices
permute_num = 10800  # number of times you want to shuffle
shuffled_data, shuffled_label = shuffle_in_unisson_permute(x_data, x_label, permute_num)

# Partition the dataset into training and validation set
NumTrainSamples = 7000  # 50% + of the data are used for training and the rest for testing
NumValidSamples = 3000
NumTestSamples = 1000
NumDim1 = x_data.shape[1]
NumDim2 = x_data.shape[2]

# .......Data for the training.......
train_images = shuffled_data[:NumTrainSamples]
# train labels
train_labels = shuffled_label[:NumTrainSamples]

sumnum = NumTrainSamples + NumValidSamples

# .......Data for the validation.......
# !!!! check that sumnum > DS[0].shape[0]
assert sumnum <= len(shuffled_data)
valid_images = shuffled_data[NumTrainSamples:sumnum]
# validation labels
valid_labels = shuffled_label[NumTrainSamples:sumnum]
# .......Data for the testing.......
assert (NumTestSamples + sumnum) <= len(shuffled_data)
test_images = shuffled_data[sumnum:NumTestSamples + sumnum]
# test labels
test_labels = shuffled_label[sumnum:NumTestSamples + sumnum]

model = tf.keras.models.Sequential()
model.add(complex_layers.ComplexInput(input_shape=(16, 8, 1)))  # Always use ComplexInput at the start
model.add(complex_layers.ComplexConv2D(8, (3, 3), activation='cart_relu'))
model.add(complex_layers.ComplexConv2D(8, (3, 3), activation='cart_relu'))
model.add(complex_layers.ComplexMaxPooling2D((2, 2)))
model.add(complex_layers.ComplexFlatten())
model.add(complex_layers.ComplexDense(8, activation='cart_relu'))
model.add(complex_layers.ComplexDropout(0.5))
model.add(complex_layers.ComplexDense(8, activation='cart_relu'))
model.add(complex_layers.ComplexDense(11, activation='convert_to_real_with_abs'))
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

# k-fold validation prevents a high variance of the validation score wrt the validation split
k = 6
num_val_samples = len(train_images) // k
num_epochs = 200
all_val_loss_histories = []
all_val_acc_histories = []
all_loss_histories = []
all_acc_histories = []
for i in range(k):
    print('processing fold #', i)
    val_data = train_images[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_labels[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(
        [train_images[:i * num_val_samples],
         train_images[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_labels[:i * num_val_samples],
         train_labels[(i + 1) * num_val_samples:]],
        axis=0)
    history = model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, validation_data=(val_data, val_targets))
    val_acc_history = history.history['val_accuracy']
    val_loss_history = history.history['val_loss']
    acc_history = history.history['accuracy']
    loss_history = history.history['loss']
    all_val_loss_histories.append(val_loss_history)
    all_val_acc_histories.append(val_acc_history)
    all_loss_histories.append(loss_history)
    all_acc_histories.append(acc_history)

average_val_acc_history = [np.mean([x[i] for x in all_val_acc_histories]) for i in range(num_epochs)]
average_val_loss_history = [np.mean([x[i] for x in all_val_loss_histories]) for i in range(num_epochs)]
average_acc_history = [np.mean([x[i] for x in all_acc_histories]) for i in range(num_epochs)]
average_loss_history = [np.mean([x[i] for x in all_loss_histories]) for i in range(num_epochs)]

iterval = 1
loss_test_array = []
acc_test_array = []
for i in range(num_epochs):
    loss_test, acc_test = model.evaluate(test_images[i*iterval:(i+1)*iterval], test_labels[i*iterval:(i+1)*iterval],
                                         verbose=2)
    loss_test_array.append(loss_test)
    acc_test_array.append(acc_test)
#model.save('kfold_SuffledRadioML2016.hdf5')
plt.figure(1)
pl1, = plt.plot(range(1, len(average_loss_history) + 1), average_loss_history, '-*')
pl2, = plt.plot(range(1, len(average_val_loss_history) + 1), average_val_loss_history, '-+')
pl3, = plt.plot(range(1, len(loss_test_array) + 1), loss_test_array, '-^')
plt.xlabel('Epochs')
plt.ylabel('Average Loss & test')
plt.title("Loss @shuffled RadioML2016, S(I)NR=" + str(snr_dB)+'dB')
plt.legend([pl1, pl2, pl3], ['Train', 'Validation', 'Test'])
plt.savefig('Plots/Loss' + str(snr_dB) + 'dBshuffledRadioML2016')
plt.show()

plt.figure(2)
pl1, = plt.plot(range(1, len(average_acc_history) + 1), average_acc_history, '-*')
pl2, = plt.plot(range(1, len(average_val_acc_history) + 1), average_val_acc_history, '-+')
pl3, = plt.plot(range(1, len(acc_test_array) + 1), acc_test_array, '-^')
plt.xlabel('Epochs')
plt.ylabel('Average Accuracy & test')
plt.title("Accuracy @shuffled RadioML2016, S(I)NR=" + str(snr_dB)+'dB')
plt.legend([pl1, pl2, pl3], ['Train', 'Validation', 'Test'])
plt.savefig('Plots/Accuracy' + str(snr_dB) + 'dBshuffledRadioML2016')
plt.show()
# smooting
def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


smooth_loss_history = smooth_curve(average_loss_history[10:])
smooth_val_loss_history = smooth_curve(average_val_loss_history[10:])
smooth_acc_history = smooth_curve(average_acc_history[10:])
smooth_val_acc_history = smooth_curve(average_val_acc_history[10:])
smooth_loss_test = smooth_curve(loss_test_array[10:])
smooth_acc_test = smooth_curve(acc_test_array[10:])

plt.figure(3)
pl1, = plt.plot(range(1, len(smooth_loss_history) + 1), smooth_loss_history)
pl2, = plt.plot(range(1, len(smooth_val_loss_history) + 1), smooth_val_loss_history)
pl3, = plt.plot(range(1, len(smooth_loss_test) + 1), smooth_loss_test)
plt.xlabel('Epochs')
plt.ylabel('Average Loss & test')
plt.title("(smoothed) Loss @shuffled RadioML2016, S(I)NR=" + str(snr_dB) + 'dB')
plt.legend([pl1, pl2, pl3], ['Train', 'Validation', 'Test'])
plt.savefig('Plots/Loss' + str(snr_dB) + 'Smooth_dBshuffledRadioML2016')
plt.show()

plt.figure(4)
pl1, = plt.plot(range(1, len(smooth_acc_history) + 1), smooth_acc_history)
pl2, = plt.plot(range(1, len(smooth_val_acc_history) + 1), smooth_val_acc_history)
pl3, = plt.plot(range(1, len(smooth_acc_test) + 1), smooth_acc_test)
plt.xlabel('Epochs')
plt.ylabel('Average Accuracy & test')
plt.title("(smoothed) Accuracy @shuffled RadioML2016, S(I)NR=" + str(snr_dB) + 'dB')
plt.legend([pl1, pl2, pl3], ['Train', 'Validation', 'Test'])
plt.savefig('Plots/Accuracy' + str(snr_dB) + 'dBshuffledRadioML2016')
plt.show()

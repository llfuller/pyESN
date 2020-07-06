import numpy as np
from matplotlib import pyplot as plt
from pyESN import ESN
import Lorenz63 as L63

# Code based on cknd's "Echo State Network as a tunable frequency generator" JuPyTer notebook.

rng = np.random.RandomState(42)

def plot_3D_orbits(num_timesteps_train, Y, Y_target):
    # Plot Y and Y_target for comparison:
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(Y_target[0, :num_timesteps_train].transpose(), Y_target[1, :num_timesteps_train].transpose(), Y_target[2, :num_timesteps_train].transpose())
    ax.plot(Y[0, :].transpose(), Y[1, :].transpose(), Y[2, :].transpose())
    plt.show()


def frequency_generator(N,min_period,max_period,n_changepoints):
    """returns a random step function with N changepoints
       and a sine wave signal that changes its frequency at
       each such step, in the limits given by min_ and max_period."""
    # vector of random indices < N, padded with 0 and N at the ends:
    changepoints = np.insert(np.sort(rng.randint(0,N,n_changepoints)),[0,n_changepoints],[0,N])
    print("changepoints: \n"+str(changepoints)+str("\n---------------------------------"))
    # list of interval boundaries between which the control sequence should be constant:
    const_intervals = list(zip(changepoints,np.roll(changepoints,-1)))[:-1]
    print("const_intervals: \n" + str(const_intervals) + str("\n---------------------------------"))
    # populate a control sequence
    frequency_control = np.zeros((N,1))
    for (t0,t1) in const_intervals:
        frequency_control[t0:t1] = rng.rand()
    print("frequency_control: \n" + str(frequency_control) + str("\n---------------------------------"))
    periods = frequency_control * (max_period - min_period) + max_period
    print("periods: \n" + str(periods) + str("\n---------------------------------"))

    # run time through a sine, while changing the period length
    frequency_output = np.zeros((N,1))
    z = 0
    for i in range(N):
        z = z + 2 * np.pi / periods[i]
        frequency_output[i] = (np.sin(z) + 1)/2
    print("frequency_output: \n" + str(frequency_output) + str("\n---------------------------------"))

    return np.hstack([np.ones((N,1)),1-frequency_control]),frequency_output

# L63.run_L63(t_final = 10000, dt = 0.01)
Y_target = np.loadtxt('L63_States.txt')
print("Shape of Y_target = "+str(np.shape(Y_target)))
# N = 15000 # signal length
# min_period = 2
# max_period = 10
# n_changepoints = int(N/200)
# frequency_control,frequency_output = frequency_generator(N,min_period,max_period,n_changepoints)
# print("frequency_control: \n" + str(frequency_control) + str("\n---------------------------------"))
# print("frequency_output: \n" + str(frequency_output) + str("\n---------------------------------"))
# print("Shape of frequency_control = "+str(np.shape(frequency_control)))
# print("Shape of frequency_output = "+str(np.shape(frequency_output)))

#
# plt.figure()
# plt.plot(frequency_control) #amplitude of frequency
# plt.figure()
# plt.plot(frequency_output) # actual displacement
# plt.show()
traintest_cutoff = 50000

train_ctrl,train_output = Y_target[:traintest_cutoff],Y_target[1:traintest_cutoff+1]
test_ctrl, test_output  = Y_target[traintest_cutoff:100000],Y_target[traintest_cutoff+1:100001]
print("Shape of train_ctrl = "+str(np.shape(train_ctrl)))
print("Shape of train_output = "+str(np.shape(train_output)))
print("Shape of test_ctrl = "+str(np.shape(test_ctrl)))
print("Shape of test_output = "+str(np.shape(test_output)))

inv_Y_target_abs_max = 1.0/abs(Y_target).max()
esn = ESN(n_inputs = 3,
          n_outputs = 3,
          n_reservoir = 400,
          spectral_radius = 0.25,
          sparsity = 0.95,
          noise = 0.0001,
          input_shift = [0,0,0],
          input_scaling = [1, 1, 1],
          teacher_scaling = 1,
          teacher_shift = 0,
          out_activation = np.tanh,
          inverse_out_activation = np.arctanh,
          random_state = rng,
          silent = False)

pred_train = esn.fit(train_ctrl,train_output)

print("test error:")
print("test ctrl"+str(test_ctrl))
pred_test = esn.predict(test_ctrl)
print(np.max(np.nan_to_num(pred_test)))
print("Shape of pred_test = "+str(np.shape(pred_test)))
print("Shape of train_ctrl = "+str(np.shape(train_ctrl)))
plot_3D_orbits(traintest_cutoff, np.nan_to_num(pred_test), train_ctrl.transpose())
print(np.sqrt(np.mean((pred_test - test_output)**2)))


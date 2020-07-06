import numpy as np
from matplotlib import pyplot as plt
from pyESN import ESN

# Code based on cknd's "Echo State Network as a tunable frequency generator" JuPyTer notebook.

rng = np.random.RandomState(42)
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


N = 15000 # signal length
min_period = 2
max_period = 10
n_changepoints = int(N/200)
frequency_control,frequency_output = frequency_generator(N,min_period,max_period,n_changepoints)
print("frequency_control: \n" + str(frequency_control) + str("\n---------------------------------"))
print("frequency_output: \n" + str(frequency_output) + str("\n---------------------------------"))
plt.figure()
plt.plot(frequency_control) #amplitude of frequency
plt.figure()
plt.plot(frequency_output) # actual displacement
plt.show()
traintest_cutoff = int(np.ceil(0.7*N))

train_ctrl,train_output = frequency_control[:traintest_cutoff],frequency_output[:traintest_cutoff]
test_ctrl, test_output  = frequency_control[traintest_cutoff:],frequency_output[traintest_cutoff:]

esn = ESN(n_inputs = 2,
          n_outputs = 1,
          n_reservoir = 200,
          spectral_radius = 0.25,
          sparsity = 0.95,
          noise = 0.001,
          input_shift = [0,0],
          input_scaling = [0.01, 3],
          teacher_scaling = 1.12,
          teacher_shift = -0.7,
          out_activation = np.tanh,
          inverse_out_activation = np.arctanh,
          random_state = rng,
          silent = False)

pred_train = esn.fit(train_ctrl,train_output)

print("test error:")
pred_test = esn.predict(test_ctrl)
plt.figure()
plt.plot(frequency_output) # target actual displacement
plt.plot(pred_test) # predicted actual displacement
plt.show()
print(np.sqrt(np.mean((pred_test - test_output)**2)))
print("max of the two arrays:" +str(np.max(frequency_control)) +","+str(np.max(frequency_output)))

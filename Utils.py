import matplotlib.pyplot as plt
import torch
import numpy as np

def plot(frame_idx, rewards):
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()

def CalcMeanRew(rew, not_dones):
    rewarr = np.asarray([line.cpu().numpy() for line in rew])
    not_donearr = np.asarray([line.cpu().numpy() for line in not_dones])
    donearr = np.ones(not_donearr.shape) - not_donearr
    rewsum = 0
    for i in range(donearr.shape[1]):
        done_ind = np.nonzero(donearr[:,i,:])
        last_done = np.amax(done_ind[0])
        rewsum += np.sum( rewarr[0:last_done,i,:] )
    mean_reward = rewsum / np.count_nonzero(donearr)
    return mean_reward

def cat_tuple_ob(states, tuple_len):
    l1 = []
    for i in range(tuple_len):
        l2 = []
        for j in range(len(states)):
            l2.append(states[j][i][:])
        l1.append(l2)
    state_l = [torch.cat(s) for s in l1]
    return state_l
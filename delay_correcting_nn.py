'''
Splits the data into obs and obs_
splits that into batches
splits that into test_train_split

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
# from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
import gym
import numpy as  np
import matplotlib.pyplot as plt

import os
import gc
import pickle
import math

#from local_remote_env import LEnv
from sklearn.model_selection import train_test_split

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

torch.cuda.empty_cache()
gc.collect()

class DCNN(nn.Module):
    def __init__(self, beta, input_dims, n_actions, layer_size, n_layers):
        super(DCNN, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions

        self.layer_size = layer_size
        self.n_layers = n_layers

        # self.fc1_dims = fc1_dims
        # self.fc2_dims = fc2_dims

        self.learning_rate = torch.tensor(beta)

        self.fc1 = nn.Linear(self.input_dims + self.n_actions , layer_size)
        self.fc2 = nn.Linear(layer_size, layer_size)
        self.fc3 = nn.Linear(layer_size, layer_size)
        self.fc4 = nn.Linear(layer_size, self.input_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        if __name__ == "__main__":
            self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=5_000_000, gamma=0.9, verbose=False) # 6.5 is roughly half 20 is /10
        else:
            self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=5_000, gamma=0.9, verbose=False) # 6.5 is roughly half 20 is /10

        self.device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        if self.n_layers > 1:
            x = self.fc2(x)
            x = F.relu(x)
            if self.n_layers > 2:
                x = self.fc3(x)
                x = F.relu(x)
        return self.fc4(x)

    def learn(self, obs, obs_): # obs includes action
        # obs = np.array(obs).astype(np.float32)
        # obs_ = np.array(obs_).astype(np.float32)

        # obs = obs.reshape(128, 1, -1)
        # obs_ = obs_.reshape(128, 1, -1)

        t_obs = torch.tensor(obs, requires_grad=True, dtype=torch.float32).to(self.device)
        t_obs_ = torch.tensor(obs_, requires_grad=True, dtype=torch.float32).to(self.device)


        prediction = self.forward(t_obs)

        self.optimizer.zero_grad()
        loss = F.huber_loss(prediction, t_obs_)
        # print(f"loss: {loss}, pred: {prediction}, actual: {obs_}")
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        loss.backward()
        self.optimizer.step()
        
        self.scheduler.step()
        # writer.add_scalar("Loss/train", loss, epoch)

        return loss.cpu().detach().item()

    def test(self, obs, obs_):
        t_obs = torch.tensor(obs, requires_grad=True, dtype=torch.float32).to(self.device)
        t_obs_ = torch.tensor(obs_, requires_grad=True, dtype=torch.float32).to(self.device)

        prediction = self.forward(t_obs)
        loss = F.mse_loss(prediction, t_obs_)
        # print("Prediction:", prediction[0], "\n\nActual: ", t_obs_[0], "\n\nDifference: ", abs(prediction[0]-t_obs_[0]))
        return loss.cpu().detach().item()


    def predict(self, obs, action):
        with torch.no_grad():
            aug_obs = torch.tensor(np.append(obs, action), device=self.device, dtype=torch.float32)
            pred = self.forward(aug_obs)
        return pred.cpu().detach().numpy()

def get_dataset(path):
    dataset = np.array([[]])

    data = np.array(pickle.load(open(path, "rb")))
    length = len(data[0])
    dataset = np.append(dataset, data)
    dataset = dataset.reshape(-1, length)
    #print(f" - loaded {path} size: {len(dataset)/int(1e6)}M")

    return dataset


# May have to fix this for HER in future
def pre_processing(env_id, obs_space, act_space, delay=1, batch_size=128):

    for item in os.listdir(f"./dataset/{env_id}/"):
    
        dataset = get_dataset(f"./dataset/{env_id}/{item}")
                
        # print(act_space, obs_space, len(dataset))
        #assert act_space + obs_space == len(dataset)

        # We trim and then add a few because of delay but this wraps round so if that happens we take away a batch
        # print(f"--- Loaded {item}: {len(dataset)/1000000}M samples")
        dataset = dataset[:(-(len(dataset) % batch_size) -batch_size + delay) ] # Trim for even batches (128 prevernts wrap arround)
        labels, _ = np.hsplit(dataset[delay:], [obs_space]) # Add delay and remove actions
        dataset = dataset[:-delay]

        # Convert into batches
        labels = labels.reshape(-1, batch_size, obs_space)
        dataset = dataset.reshape(-1, batch_size, obs_space+act_space)

        X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, shuffle=True)
        
        yield X_train, X_test, y_train, y_test

def train(X_train,X_test, y_train, y_test, model, obs_space, act_space, batch_size=256, env_id=("manipulator", "bring_ball"), delay=8):
	# print("before",len(X_train))
    losses = []

    for i in range(len(X_train)):
        losses.append(model.learn(X_train[i], y_train[i]))

    test_loss = test(X_test, y_test, test_model=model, env_id=env_id, delay=delay)

    return model, float(test_loss), losses


def test(X_test, y_test, test_model, env_id, delay):
    if not test_model: # test_model not None
        test_model = DCNN(**pickle.load(open(f"./models/{env_id}/{delay}_step_params.pickle", "rb")))
        state_dict = torch.load(f"./models/{env_id}/{delay}_step_prediction_sd.pt")
        test_model.load_state_dict(state_dict)

    test_model.eval()

    with torch.no_grad():
        loss = test_model.test(X_test, y_test)

    test_model.train()
    return float(loss)

def plot(stats, plotdir, label=None):
    for key, value in stats.items():
        plt.plot(range(len(value)), value,label=label)
        plt.ylabel(key)
        plt.xlabel("training steps")
        # plt.savefig(plotdir + key +".png")


def run(env_id, epochs, batch_size, file_name):
    delay = 1
    PLOT_DIR = "./plots/"

    #print(env_id)
    # import sys
    # sys.exit()
    env = gym.make(env_id)
    act_space = env.action_space.shape[0]
    obs_space = env.observation_space.shape[0]

    # Test params
    for n_layers in [2]:
        for layer_size in [128]:

            losses = np.array([])
            test_losses = np.array([])
            epoch_losses = []
            episodic_losses = []
            obs_batch = []
            new_obs_batch = []

            print(f"\n\nn_layers: {n_layers} layer_size: {layer_size}")
            params = dict(beta=0.00005, input_dims=obs_space, n_actions=act_space, layer_size=layer_size, n_layers=n_layers)
            model = DCNN(**params)
            if not os.path.isdir(f"./models/{env_id}"):
                os.mkdir(f"./models/{env_id}")
            pickle.dump(params, open(f"./models/{env_id}/{delay}_step_params.pickle", "wb"))

            for e in range(epochs):
                for X_train, X_test, y_train, y_test in pre_processing(env_id, obs_space, act_space, delay, batch_size):
                    model, test_loss, loss = train(X_train, X_test, y_train, y_test, model, obs_space=obs_space, act_space=act_space, batch_size=batch_size, env_id=env_id, delay=delay)
                    test_losses = np.append(test_losses, test_loss)
                    losses = np.append(losses, loss)

                    # print(f"--- Loss: {np.mean(loss)} Test Loss: {test_loss}")
                    epoch_losses.append(np.mean(losses[-len(losses)//(e+1):]))
                    del test_loss
                    del loss
                print(f"Finished Epoch {e} avg loss: {epoch_losses[-1]} avg test_loss: {np.mean(test_losses[-len(test_losses)//(e+1):])}")

            torch.save(model.state_dict(), file_name)
            # plot({"loss":losses, "test loss":test_losses,"epoch loss": epoch_losses}, plotdir=PLOT_DIR, label=f"{n_layers}-{layer_size}")
    # plt.savefig(PLOT_DIR + "key" +".png")

if __name__== "__main__":
    env_id = "FetchPush-RemotePDNorm-v0"
    epochs = 3
    batch_size = 256
    n_layers = 2
    layer_size = 128
    delay = 1
    file_name = f"./models/{env_id}/{n_layers}-{layer_size}-{delay}_step_prediction_sd.pt"
    run(env_id,  epochs, batch_size, file_name)

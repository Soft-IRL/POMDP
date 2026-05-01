import pickle as pkl 
import numpy as np

losses = pkl.load(open("light_dark_model_reconstruction_losses.pkl", "rb"))
loss_mse = losses[0]
loss_kl  = losses[1]
print(len(loss_mse))
print(len(loss_kl))
print(loss_mse[0], loss_mse[-1])
print(loss_kl[0], loss_kl[-1])
#print(np.mean(losses[-1]))
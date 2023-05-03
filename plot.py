import sys
import matplotlib.pyplot as plt

steps, returns, A_loss, C_loss, E_loss, KL, clip = [], [], [], [], [], [], []
filename = sys.argv[1]
with open(filename) as f:
	for line in f:
		line = line.strip().split()
		steps.append(  line[1])
		returns.append(line[3])
		A_loss.append( line[5])
		C_loss.append( line[7])
		E_loss.append( line[9])
		KL.append(     line[11])
		clip.append(   line[13])
fig, axs = plt.subplots(3, 2)
axs[0, 0].plot(steps, returns)
axs[0, 0].set_title('Returns')
axs[0, 0].set(xlabel='Steps', ylabel='Returns')
axs[0, 1].plot(steps, A_loss)
axs[0, 1].set_title('Actor loss')
axs[0, 1].set(xlabel='Steps', ylabel='Loss')
axs[1, 1].plot(steps, C_loss)
axs[1, 1].set_title('Critic loss')
axs[1, 1].set(xlabel='Steps', ylabel='Loss')
axs[2, 1].plot(steps, E_loss)
axs[2, 1].set_title('Entropy loss')
axs[2, 1].set(xlabel='Steps', ylabel='Loss')
axs[1, 0].plot(steps, KL)
axs[1, 0].set_title('KL aproximation')
axs[1, 0].set(xlabel='Steps', ylabel='KL')
axs[2, 0].plot(steps, clip)
axs[2, 0].set_title('Clip fraction')
axs[2, 0].set(xlabel='Steps', ylabel='Fraction')
plt.show()

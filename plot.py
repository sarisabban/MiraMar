import datetime
import matplotlib.pyplot as plt

updates, steps, returns, SD, A_loss, C_loss, E_loss, loss, KL, clip, var = [], [], [], [], [], [], [], [], [], [], []
filename = 'train.log'
with open(filename) as f:
	for skip in range(18): next(f)
	for line in f:
		line = line.strip().split()
		updates.append(int(line[1].replace(',', '')))
		steps.append(  int(line[3].replace(',', '')))
		returns.append(float(line[5].replace(',', '')))
		SD.append(     float(line[7].replace(',', '')))
		A_loss.append( float(line[9].replace(',', '')))
		C_loss.append( float(line[11].replace(',', '')))
		E_loss.append( float(line[14].replace(',', '')))
		loss.append(   float(line[17].replace(',', '')))
		KL.append(     float(line[19].replace(',', '')))
		clip.append(   float(line[21].replace(',', '')))
		var.append(    float(line[24].replace(',', '')))

fig, axs = plt.subplots(4, 2)
axs[0, 0].plot(steps, returns)
axs[0, 0].set_title('Returns')
axs[0, 0].set(xlabel='Steps', ylabel='Returns')

axs[0, 1].plot(updates, loss)
axs[0, 1].set_title('Final Loss')
axs[0, 1].set(xlabel='Updates', ylabel='Loss')

axs[1, 1].plot(updates, A_loss)
axs[1, 1].set_title('Actor loss')
axs[1, 1].set(xlabel='Updates', ylabel='Loss')

axs[2, 1].plot(updates, C_loss)
axs[2, 1].set_title('Critic loss')
axs[2, 1].set(xlabel='Updates', ylabel='Loss')

axs[3, 1].plot(updates, E_loss)
axs[3, 1].set_title('Entropy loss')
axs[3, 1].set(xlabel='Updates', ylabel='Loss')

axs[1, 0].plot(updates, KL)
axs[1, 0].set_title('KL aproximation')
axs[1, 0].set(xlabel='Updates', ylabel='Ratio')

axs[2, 0].plot(updates, clip)
axs[2, 0].set_title('Clip fraction')
axs[2, 0].set(xlabel='Updates', ylabel='Fraction')

axs[3, 0].plot(updates, var)
axs[3, 0].set_title('Explained variance')
axs[3, 0].set(xlabel='Updates', ylabel='variance')

plt.show()

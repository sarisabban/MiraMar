import datetime
import matplotlib.pyplot as plt

updates, steps, returns, SD, A_loss, C_loss, E_loss, loss, KL, clip, time = [], [], [], [], [], [], [], [], [], [], []
filename = 'train.log'
with open(filename) as f:
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
		h = int(line[24:][-1].split(':')[0])
		m = int(line[24:][-1].split(':')[1])
		s = int(line[24:][-1].split(':')[2])
		if line[24:][1] == 'days,': d = int(line[24:][0])
		else:                       d = 0
		T = datetime.timedelta(days=d, hours=h, minutes=m, seconds=s)
		T = T.total_seconds()
		time.append(T)

fig, axs = plt.subplots(4, 2)
axs[0, 0].plot(steps, returns)
axs[0, 0].set_title('Returns')
axs[0, 0].set(xlabel='Steps', ylabel='Returns')

axs[0, 1].plot(steps, loss)
axs[0, 1].set_title('Final Loss')
axs[0, 1].set(xlabel='Steps', ylabel='Loss')

axs[1, 1].plot(steps, A_loss)
axs[1, 1].set_title('Actor loss')
axs[1, 1].set(xlabel='Steps', ylabel='Loss')

axs[2, 1].plot(steps, C_loss)
axs[2, 1].set_title('Critic loss')
axs[2, 1].set(xlabel='Steps', ylabel='Loss')

axs[3, 1].plot(steps, E_loss)
axs[3, 1].set_title('Entropy loss')
axs[3, 1].set(xlabel='Steps', ylabel='Loss')

axs[1, 0].plot(steps, KL)
axs[1, 0].set_title('KL aproximation')
axs[1, 0].set(xlabel='Steps', ylabel='KL')

axs[2, 0].plot(steps, clip)
axs[2, 0].set_title('Clip fraction')
axs[2, 0].set(xlabel='Steps', ylabel='Fraction')

axs[3, 0].plot(steps, time)
axs[3, 0].set_title('Time')
axs[3, 0].set(xlabel='Steps', ylabel='Seconds')

plt.show()

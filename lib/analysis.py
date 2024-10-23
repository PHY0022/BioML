import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

def plotAAC(posAAC, negAAC):
    # fig, ax = plt.subplots()
    pos = np.arange(len(posAAC.index)) * 2
    width = 0.45
    padding = width + 0.15

    plt.bar(pos          , posAAC['freq'] * 100.0, width, label='Positive', color='r')
    plt.bar(pos + padding, negAAC['freq'] * 100.0, width, label='Negative', color='tab:blue')
    plt.xticks(pos + padding / 2, posAAC.index)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=100))
    plt.xlabel('Amino Acid')
    plt.ylabel('Frequency')
    plt.legend(loc='upper left')

    plt.show()

def plotPWM(PWM):
    plt.figure(figsize=(12,6))
    plt.imshow(PWM, cmap='hot', interpolation='nearest')
    plt.xlabel('Position')
    plt.ylabel('Amino Acid')
    plt.xticks(np.arange(PWM.shape[1]), PWM.columns)
    plt.yticks(np.arange(PWM.shape[0]), PWM.index)
    plt.colorbar()
    plt.show()
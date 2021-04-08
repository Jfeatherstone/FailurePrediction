import numpy as np
import matplotlib.pyplot as plt


# This is just to take a look at a given sample
# Designed to be used directly with the output of randomSample
# Arguments are setup so that you can provide the outout of randomSample directly
# or can specify each array individually
def visualizeSample(sample=None, tArr=np.array([]), sArr=np.array([]), fArr=np.array([]), events=np.array([]), savePath=None):

    if not tArr.any():
        tArr, sArr, fArr, events = sample

    fig, ax1 = plt.subplots()
    
    # I wanted the colors to look pretty so I used some that aren't used very often
    # (Don't worry, I've used my colorblindness simulator to make sure they are
    # distinguishable!)
    ax1Color = 'darkcyan'
    ax1.plot(tArr, fArr, color=ax1Color)
    ax1.set_ylabel('Force Sensor Reading', color=ax1Color)
    ax1.tick_params(axis='y', labelcolor=ax1Color)

    ax2Color = 'indigo'
    ax2 = ax1.twinx()
    ax2.plot(tArr, sArr, color=ax2Color)
    ax2.tick_params(axis='y', labelcolor=ax2Color)
    ax2.set_ylabel('Score', color=ax2Color)

    for time in [tArr[i] for i in range(len(tArr)) if events[i] != 0]:
        ax1.axvline(time, linestyle='--', color='tab:gray', alpha=.8)

    ax1.set_xlim([tArr[0], tArr[-1]])
    if savePath != None:
        plt.savefig(savePath)

    fig.patch.set_facecolor('white')
    plt.show()

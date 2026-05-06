def plot_correlation(z_vec, x0, x1, ax, lag=20., title='firing_rate vs LFP'):
    ''' mls
    on axes plot the correlation between x0 and x1
    
    args:
    ::
        x0 : first dataset
        x1 : second dataset - e.g., the multichannel LFP
        ax : matplotlib.axes.AxesSubplot object
        title : text to be used as current axis object title
    '''
    zvec = np.r_[z_vec]
    zvec = np.r_[zvec, zvec[-1] + np.diff(zvec)[-1]]
    
    xcorr_all=np.zeros((np.size(z_vec), x0.shape[0]))
    for i, z in enumerate(z_vec):
        x2 = x1[i, ]
        xcorr1 = np.correlate(normalize(x0),
                              normalize(x2), 'same') / x0.size
        xcorr_all[i,:]=xcorr1  

    # Find limits for the plot
    vlim = abs(xcorr_all).max()
    vlimround = 2.**np.round(np.log2(vlim))
    
    yticklabels=[]
    yticks = []
    ylimfound=np.zeros((1,2))
    for i, z in enumerate(z_vec):
        ind = np.arange(x0.size) - x0.size/2
        ax.plot(ind, xcorr_all[i,::-1] * 100. / vlimround + z, 'k',
                clip_on=True, rasterized=False)
        yticklabels.append('ch. %i' %(i+1))
        yticks.append(z)    

    remove_axis_junk(ax)
    ax.set_title(title)
    ax.set_xlabel(r'lag $\tau$ (ms)')

    ax.set_xlim(-lag, lag)
    ax.set_ylim(z-100, 100)
    
    axis = ax.axis()
    ax.vlines(0, axis[2], axis[3], 'r', 'dotted')

    ax.yaxis.set_ticks(yticks)
    ax.yaxis.set_ticklabels(yticklabels)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # Create a scaling bar
    ax.plot([lag, lag],
        [0, 100], lw=2, color='k', clip_on=False)
    ax.text(lag, 50, r'CC=%.2f' % vlimround,
            rotation='vertical', va='center')
def plot_signal_sum(ax, z, fname='LFPsum.h5', unit='mV',
                    ylabels=True, scalebar=True, vlimround=None,
                    T=[0, 1000], color='k',
                    label=''):
    '''
    on axes plot the signal contributions
    
    args:
    ::
        ax : matplotlib.axes.AxesSubplot object
        z : np.ndarray
        T : list, [tstart, tstop], which timeinterval
        ylims : list, set range of yaxis to scale with other plots
        fancy : bool, 
        scaling_factor : float, scaling factor (e.g. to scale 10% data set up)
    '''    
    #open file and get data, samplingrate
    f = h5py.File(fname)
    data = f['data'].value
    dataT = data.T - data.mean(axis=1)
    data = dataT.T
    srate = f['srate'].value
    
    #close file object
    f.close()

    # normalize data for plot
    tvec = np.arange(data.shape[1]) * 1000. / srate
    slica = (tvec <= T[1]) & (tvec >= T[0])
    zvec = np.r_[z]
    zvec = np.r_[zvec, zvec[-1] + np.diff(zvec)[-1]]
    vlim = abs(data[:, slica]).max()
    if vlimround is None:
        vlimround = 2.**np.round(np.log2(vlim))
    yticklabels=[]
    yticks = []
    
    colors = [color]*data.shape[0]
    
    for i, z in enumerate(z):
        if i == 0:
            ax.plot(tvec[slica], data[i, slica] * 100 / vlimround + z,
                    color=colors[i], rasterized=False, label=label,
                    clip_on=False)
        else: 
            ax.plot(tvec[slica], data[i, slica] * 100 / vlimround + z,
                    color=colors[i], rasterized=False, clip_on=False)
        yticklabels.append('ch. %i' % (i+1))
        yticks.append(z)
     
    if scalebar:
        ax.plot([tvec[slica][-1], tvec[slica][-1]],
                [-0, -100], lw=2, color='k', clip_on=False)
        ax.text(tvec[slica][-1]+np.diff(T)*0.02, -50,
                r'%g %s' % (vlimround, unit),
                color='k', rotation='vertical')

    ax.axis(ax.axis('tight'))
    ax.yaxis.set_ticks(yticks)
    if ylabels:
        ax.yaxis.set_ticklabels(yticklabels)
    else:
        ax.yaxis.set_ticklabels([])

    for loc, spine in ax.spines.items():
        if loc in ['right', 'top']:
            spine.set_color('none')            
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xlabel(r'time (ms)', labelpad=0)
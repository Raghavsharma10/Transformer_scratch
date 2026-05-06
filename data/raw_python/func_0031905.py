def plot_signal_power_colorplot(ax, params, fname, transient=200, Df=None,
                                mlab=True, NFFT=1000,
                                window=plt.mlab.window_hanning,
                                noverlap=0,
                                cmap = plt.cm.get_cmap('jet', 21),
                                vmin=None,
                                vmax=None):
    '''
    on axes plot the LFP power spectral density  
    The whole signal duration is used.
    args:
    ::
        ax : matplotlib.axes.AxesSubplot object
        fancy : bool, 
    '''
  
    zvec = np.r_[params.electrodeParams['z']]
    zvec = np.r_[zvec, zvec[-1] + np.diff(zvec)[-1]]
    
    #labels
    yticklabels=[]
    yticks = []

    for i, kk in enumerate(params.electrodeParams['z']):
        yticklabels.append('ch. %i' % (i+1))
        yticks.append(kk)

    
    freqs, PSD = calc_signal_power(params, fname=fname, transient=transient,Df=Df,
                                   mlab=mlab, NFFT=NFFT,
                                   window=window, noverlap=noverlap)

    #plot only above 1 Hz
    inds = freqs >= 1  # frequencies greater than 4 Hz  
    im = ax.pcolormesh(freqs[inds], zvec+50, PSD[:, inds],
                       rasterized=True, norm=LogNorm(),
                       vmin=vmin,vmax=vmax,
                       cmap=cmap, )
    
    ax.yaxis.set_ticks(yticks)
    ax.yaxis.set_ticklabels(yticklabels)
    ax.semilogx()
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xlabel(r'$f$ (Hz)', labelpad=0.1)
    ax.axis(ax.axis('tight'))


    return im
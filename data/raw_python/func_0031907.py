def plotting_correlation(params, x0, x1, ax, lag=20., scaling=None, normalize=True,
                         color='k', unit=r'$cc=%.3f$' , title='firing_rate vs LFP',
                         scalebar=True, **kwargs):
    ''' mls
    on axes plot the correlation between x0 and x1
    
    args:
    ::
        x0 : first dataset
        x1 : second dataset - the LFP usually here
        ax : matplotlib.axes.AxesSubplot object
        title : text to be used as current axis object title
        normalize : if True, signals are z-scored before applying np.correlate
        unit : unit for scalebar
    '''
    zvec = np.r_[params.electrodeParams['z']]
    zvec = np.r_[zvec, zvec[-1] + np.diff(zvec)[-1]]
    
    xcorr_all=np.zeros((params.electrodeParams['z'].size, x0.shape[-1]))
    
    if normalize:
        for i, z in enumerate(params.electrodeParams['z']):
            if x0.ndim == 1:
                x2 = x1[i, ]
                xcorr1 = np.correlate(helpers.normalize(x0),
                                      helpers.normalize(x2), 'same') / x0.size
            elif x0.ndim == 2:
                xcorr1 = np.correlate(helpers.normalize(x0[i, ]),
                                      helpers.normalize(x1[i, ]), 'same') / x0.shape[-1]
            
            xcorr_all[i,:]=xcorr1
    else:
        for i, z in enumerate(params.electrodeParams['z']):
            if x0.ndim == 1:
                x2 = x1[i, ]
                xcorr1 = np.correlate(x0,x2, 'same')
            elif x0.ndim == 2:
                xcorr1 = np.correlate(x0[i, ],x1[i, ], 'same')
                
            xcorr_all[i,:]=xcorr1
    

    # Find limits for the plot
    if scaling is None:
        vlim = abs(xcorr_all).max()
        vlimround = 2.**np.round(np.log2(vlim))
    else:
        vlimround = scaling
    
    yticklabels=[]
    yticks = []
    
    #temporal slicing
    lagvector = np.arange(-lag, lag+1).astype(int)
    inds = lagvector + x0.shape[-1] / 2
    
    
    for i, z in enumerate(params.electrodeParams['z']):
        ax.plot(lagvector, xcorr_all[i,inds[::-1]] * 100. / vlimround + z, 'k',
                clip_on=True, rasterized=False, color=color, **kwargs)
        yticklabels.append('ch. %i' %(i+1))
        yticks.append(z)    

    phlp.remove_axis_junk(ax)
    ax.set_title(title, va='center')
    ax.set_xlabel(r'$\tau$ (ms)', labelpad=0.1)

    ax.set_xlim(-lag, lag)
    ax.set_ylim(z-100, 100)
    
    axis = ax.axis()
    ax.vlines(0, axis[2], axis[3], 'k' if analysis_params.bw else 'k', 'dotted', lw=0.25)

    ax.yaxis.set_ticks(yticks)
    ax.yaxis.set_ticklabels(yticklabels)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ## Create a scaling bar
    if scalebar:
        ax.plot([lag, lag],
            [-1500, -1400], lw=2, color='k', clip_on=False)
        ax.text(lag*1.04, -1450, unit % vlimround,
                    rotation='vertical', va='center')
    
    return xcorr_all[:, inds[::-1]], vlimround
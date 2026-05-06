def plot_signal_sum_colorplot(ax, params, fname='LFPsum.h5', unit='mV', N=1, ylabels = True,
                              T=[800, 1000], ylim=[-1500, 0], fancy=False, colorbar=True,
                              cmap='spectral_r', absmax=None, transient=200, rasterized=True):
    '''
    on colorplot and as background plot the summed CSD contributions
    
    args:
    ::
        ax : matplotlib.axes.AxesSubplot object
        T : list, [tstart, tstop], which timeinterval
        ylims : list, set range of yaxis to scale with other plots
        fancy : bool, 
        N : integer, set to number of LFP generators in order to get the normalized signal
    '''
    f = h5py.File(fname)
    data = f['data'].value
    tvec = np.arange(data.shape[1]) * 1000. / f['srate'].value
    
    #for mean subtraction
    datameanaxis1 = f['data'].value[:, tvec >= transient].mean(axis=1)
    
    # slice
    slica = (tvec <= T[1]) & (tvec >= T[0])
    data = data[:,slica]

    # subtract mean
    #dataT = data.T - data.mean(axis=1)
    dataT = data.T - datameanaxis1
    data = dataT.T

    # normalize
    data = data/N
    zvec = params.electrodeParams['z']
    
    if fancy:
        colors = phlp.get_colors(data.shape[0])
    else:
        colors = ['k']*data.shape[0]
    
    if absmax == None:
        absmax=abs(np.array([data.max(), data.min()])).max()  
    im = ax.pcolormesh(tvec[slica], np.r_[zvec, zvec[-1] + np.diff(zvec)[-1]] + 50, data,
                           rasterized=rasterized, vmax=absmax, vmin=-absmax, cmap=cmap)
    ax.set_yticks(params.electrodeParams['z'])
    if ylabels:
        yticklabels = ['ch. %i' %(i+1) for i in np.arange(len(params.electrodeParams['z']))]
        ax.set_yticklabels(yticklabels)
    else:
        ax.set_yticklabels([])

    if colorbar:
        #colorbar
        divider=make_axes_locatable(ax)
        cax=divider.append_axes("right", size="5%", pad=0.1)
        cbar=plt.colorbar(im, cax=cax)
        cbar.set_label(unit,labelpad=0.1)
        
    plt.axis('tight')

    ax.set_ylim(ylim)

    
    f.close()
    
    return im
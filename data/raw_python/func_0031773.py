def fig_kernel_lfp_CINPLA(savefolders, params, transient=200, X='L5E', lags=[20, 20]):

    '''
    This function calculates the  STA of LFP, extracts kernels and recontructs the LFP from kernels.
    
    kwargs:
    ::
      transient : the time in milliseconds, after which the analysis should begin
                so as to avoid any starting transients
      X : id of presynaptic trigger population
       
    '''

    # Electrode geometry
    zvec = np.r_[params.electrodeParams['z']]

    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'



    ana_params.set_PLOS_2column_fig_style(ratio=0.5)
    # Start the figure
    fig = plt.figure()
    fig.subplots_adjust(left=0.06, right=0.95, bottom=0.075, top=0.925, hspace=0.23, wspace=0.55) 

    # create grid_spec
    gs = gridspec.GridSpec(1, 7)


    ###########################################################################
    # Part A: spikegen "network" activity
    ############################################################################
        
    # path to simulation files
    params.savefolder = 'simulation_output_spikegen'
    params.figures_path = os.path.join(params.savefolder, 'figures')
    params.spike_output_path = os.path.join(params.savefolder,
                                                       'processed_nest_output')
    params.networkSimParams['spike_output_path'] = params.spike_output_path
    
    
    # Get the spikegen LFP:
    f = h5py.File(os.path.join('simulation_output_spikegen', 'LFPsum.h5'))
    srate = f['srate'].value
    tvec = np.arange(f['data'].shape[1]) * 1000. / srate
    
    # slice
    inds = (tvec < params.tstop) & (tvec >= transient)
    
    data_sg_raw = f['data'].value.astype(float)
    f.close()
    #
    # kernel width
    kwidth = 20
    
    # extract kernels
    kernels = np.zeros((len(params.N_X), 16, 100))
    for j in range(len(params.X)):
        kernels[j] = data_sg_raw[:, 100+kwidth+j*100:100+kwidth+(j+1)*100] / params.N_X[j]
    
    
    # create some dummy spike times
    activationtimes = np.array([x*100 for x in range(3,11)] + [200])
    networkSimSpikegen = CachedNetwork(**params.networkSimParams)

    x, y = networkSimSpikegen.get_xy([transient, params.tstop])

    ###########################################################################
    # Part A: spatiotemporal kernels, all presynaptic populations
    ############################################################################
    
    titles = ['TC',
              'L23E/I',
              'LFP kernels \n L4E/I',
              'L5E/I',
              'L6E/I',
              ]
    
    COUNTER = 0 
    for i, X__ in enumerate(([['TC']]) + zip(params.X[1::2], params.X[2::2])):        
        ax = fig.add_subplot(gs[0, i])
        if i == 0:
            phlp.annotate_subplot(ax, ncols=7, nrows=4, letter=alphabet[0], linear_offset=0.02)
    
        for j, X_ in enumerate(X__):
            # create spikegen histogram for population Y
            cinds = np.arange(activationtimes[np.arange(-1, 8)][COUNTER]-kwidth,
                              activationtimes[np.arange(-1, 8)][COUNTER]+kwidth+2)
            x0_sg = np.histogram(x[X_], bins=tvec[cinds])[0].astype(float)
        
            if X_ == ('TC'):
                color='r'
            else:
                color=('r', 'b')[j]
            
            
            # plot kernel as correlation of spikegen LFP signal with delta spike train
            xcorr, vlimround = plotting_correlation(x0_sg/x0_sg.sum()**2, 
                                 data_sg_raw[:, cinds[:-1]]*1E3,
                                 ax, normalize=False,
                                 lag=kwidth,
                                 color=color,
                                 scalebar=False)
            if i > 0:
                ax.set_yticklabels([])
            
            ## Create scale bar
            ax.plot([kwidth, kwidth],
                [-1500 + j*3*100, -1400 + j*3*100], lw=2, color=color,
                clip_on=False)
            ax.text(kwidth*1.08, -1450 + j*3*100, '%.1f $\mu$V' % vlimround,
                        rotation='vertical', va='center')
    
            ax.set_xlim((-5, kwidth))
            ax.set_xticks([-20, 0, 20])
            ax.set_xticklabels([-20, 0, 20])
            
            COUNTER += 1
            
        ax.set_title(titles[i])


    for i, (savefolder, lag) in enumerate(zip(savefolders, lags)):
        
        # path to simulation files
        params.savefolder = os.path.join(os.path.split(params.savefolder)[0],
                                         savefolder)
        params.figures_path = os.path.join(params.savefolder, 'figures')
        params.spike_output_path = os.path.join(params.savefolder,
                                                'processed_nest_output')
        params.networkSimParams['spike_output_path'] = params.spike_output_path

        #load spike as database inside function to avoid buggy behaviour
        networkSim = CachedNetwork(**params.networkSimParams)

    
    
        # Get the Compound LFP: LFPsum : data[nchannels, timepoints ]
        f = h5py.File(os.path.join(params.savefolder, 'LFPsum.h5'))
        data_raw = f['data'].value
        srate = f['srate'].value
        tvec = np.arange(data_raw.shape[1]) * 1000. / srate
        # slice
        inds = (tvec < params.tstop) & (tvec >= transient)
        data = data_raw[:,inds]
        # subtract mean
        dataT = data.T - data.mean(axis=1)
        data = dataT.T
        f.close()
    
        # Get the spikegen LFP:
        f = h5py.File(os.path.join('simulation_output_spikegen', 'LFPsum.h5'))
        data_sg_raw = f['data'].value
        # slice
        data_sg = data_sg_raw[:,inds[data_sg_raw.shape[1]]]
        f.close()
    
    
    
    
        ########################################################################
        # Part B: STA LFP
        ########################################################################
        
        ax = fig.add_subplot(gs[0, 5 + i])
        phlp.annotate_subplot(ax, ncols=15, nrows=4, letter=alphabet[i+1],
                              linear_offset=0.02)
          
        # collect the spikes x is the times, y is the id of the cell.
        x, y = networkSim.get_xy([0,params.tstop])
        
        # Get the spikes for the population of interest given as 'Y'
        bins = np.arange(0, params.tstop+2)
        x0_raw = np.histogram(x[X], bins=bins)[0]
        x0 = x0_raw[inds].astype(float)
    
        # correlation between firing rate and LFP deviation
        # from mean normalized by the number of spikes  
        xcorr, vlimround = plotting_correlation(x0/x0.sum(), 
                             data*1E3,
                             ax, normalize=False, 
                             #unit='%.3f mV',
                             lag=lag,
                             scalebar=False,
                             color='k',
                             title='stLFP\n(trigger %s)' %X,
                             )

        # Create scale bar
        ax.plot([lag, lag],
            [-1500, -1400], lw=2, color='k',
            clip_on=False)
        ax.text(lag*1.04, -1450, '%.1f $\mu$V' % vlimround,
                    rotation='vertical', va='center')


        [Xind] = np.where(np.array(networkSim.X) == X)[0]        
                
        # create spikegen histogram for population Y
        x0_sg = np.zeros(x0.shape, dtype=float)
        x0_sg[activationtimes[Xind]] += params.N_X[Xind]
        
        ax.set_yticklabels([])
        ax.set_xticks([-lag, 0, lag])
        ax.set_xticklabels([-lag, 0, lag])
        

    return fig
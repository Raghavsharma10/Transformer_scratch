def plotPowers(ax, params, popkeys, dataset, linestyles, linewidths, transient=200, SCALING_POSTFIX='', markerstyles=None):
    '''plot power (variance) as function of depth for total and separate
    contributors

    Plot variance of sum signal
    '''
    
    colors = phlp.get_colors(len(popkeys))
    
    depth = params.electrodeParams['z']
    zpos = np.r_[params.layerBoundaries[:, 0],
                 params.layerBoundaries[-1, 1]]

    for i, layer in enumerate(popkeys):
        f = h5py.File(os.path.join(params.populations_path,
                                   '%s_population_%s' % (layer, dataset) + SCALING_POSTFIX + '.h5' ))
        ax.semilogx(f['data'].value[:, transient:].var(axis=1), depth,
                 color=colors[i],
                 ls=linestyles[i],
                 lw=linewidths[i],
                 marker=None if markerstyles is None else markerstyles[i],
                 markersize=2.5,
                 markerfacecolor=colors[i],
                 markeredgecolor=colors[i],
                 label=layer,
                 clip_on=True
                 )
    
        f.close()
    
    f = h5py.File(os.path.join(params.savefolder, '%ssum' % dataset + SCALING_POSTFIX + '.h5' ))
    ax.plot(f['data'].value[:, transient:].var(axis=1), depth,
                 'k', label='SUM', lw=1.25, clip_on=False)
    
    f.close()

    ax.set_yticks(zpos)
    ax.set_yticklabels([])
    #ax.set_xscale('log')
    try: # numticks arg only exists for latest matplotlib version
        ax.xaxis.set_major_locator(plt.LogLocator(base=10,
                                    subs=np.linspace(-10, 10, 2), numticks=6))
    except:
        ax.xaxis.set_major_locator(plt.LogLocator(base=10,
                                    subs=np.linspace(-10, 10, 2)))
    ax.xaxis.set_minor_locator(plt.LogLocator(base=10, subs=[1.]))
    ax.axis('tight')
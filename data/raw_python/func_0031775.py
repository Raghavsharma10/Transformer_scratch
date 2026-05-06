def plot_sim(ax, cell, synapse, grid_electrode, point_electrode, letter='a'):
    '''create a plot'''
    
    fig = plt.figure(figsize = (3.27*2/3, 3.27*2/3))
    
    ax = fig.add_axes([.1,.05,.9,.9], aspect='equal', frameon=False)
    
    phlp.annotate_subplot(ax, ncols=1, nrows=1, letter=letter, fontsize=16)
    
    cax = fig.add_axes([0.8, 0.2, 0.02, 0.2], frameon=False)
    
    
    LFP = np.max(np.abs(grid_electrode.LFP),1).reshape(X.shape)
    im = ax.contour(X, Z, np.log10(LFP), 
               50,
               cmap='RdBu',
               linewidths=1.5,
               zorder=-2)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('$|\phi(\mathbf{r}, t)|_\mathrm{max}$ (nV)')
    cbar.outline.set_visible(False)
    #get some log-linear tickmarks and ticklabels
    ticks = np.arange(np.ceil(np.log10(LFP.min())), np.ceil(np.log10(LFP.max())))
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(10.**ticks * 1E6) #mv -> nV
    
    zips = []
    for x, z in cell.get_idx_polygons():
        zips.append(zip(x, z))
    polycol = PolyCollection(zips,
                             edgecolors='k',
                             linewidths=0.5,
                             facecolors='k')
    ax.add_collection(polycol)
    
    ax.plot([100, 200], [-400, -400], 'k', lw=1, clip_on=False)
    ax.text(150, -470, r'100$\mu$m', va='center', ha='center')
    
    ax.axis('off')
    
    
    ax.plot(cell.xmid[cell.synidx],cell.zmid[cell.synidx], 'o', ms=5,
            markeredgecolor='k',
            markerfacecolor='r')
    
    color_vec = ['blue','green']
    for i in xrange(2):
        ax.plot(point_electrode_parameters['x'][i],
                        point_electrode_parameters['z'][i],'o',ms=6,
                        markeredgecolor='none',
                        markerfacecolor=color_vec[i])
    
    
    plt.axes([.11, .075, .25, .2])
    plt.plot(cell.tvec,point_electrode.LFP[0]*1e6,color=color_vec[0], clip_on=False)
    plt.plot(cell.tvec,point_electrode.LFP[1]*1e6,color=color_vec[1], clip_on=False)
    plt.axis('tight')
    ax = plt.gca()
    ax.set_ylabel(r'$\phi(\mathbf{r}, t)$ (nV)') #rotation='horizontal')
    ax.set_xlabel('$t$ (ms)', va='center')
    for loc, spine in ax.spines.iteritems():
        if loc in ['right', 'top']:
            spine.set_color('none')            
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
    plt.axes([.11, 0.285, .25, .2])
    plt.plot(cell.tvec,synapse.i*1E3, color='red', clip_on=False)
    plt.axis('tight')
    ax = plt.gca()
    ax.set_ylabel(r'$I_{i, j}(t)$ (pA)', ha='center', va='center') #, rotation='horizontal')
    for loc, spine in ax.spines.iteritems():
        if loc in ['right', 'top']:
            spine.set_color('none')            
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xticklabels([])

    return fig
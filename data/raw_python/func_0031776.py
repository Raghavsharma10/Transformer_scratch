def plot_sim_tstep(fig, ax, cell, synapse, grid_electrode, point_electrode, tstep=0,
                   letter='a',title='', cbar=True, show_legend=False):
    '''create a plot'''
    ax.set_title(title)
    
    if letter != None:
        phlp.annotate_subplot(ax, ncols=3, nrows=1, letter=letter, linear_offset=0.05, fontsize=16)

    
    
    LFP = grid_electrode.LFP[:, tstep].reshape(X.shape).copy()
    LFP *= 1E6 #mv -> nV
    vlim = 50
    levels = np.linspace(-vlim*2, vlim*2, 401)
    cbarticks = np.mgrid[-50:51:20]
    #cbarticks = [-10**np.floor(np.log10(vlim)),
    #             0,
    #             10**np.floor(np.log10(vlim)),]
    #force dashed for negative values
    linestyles = []
    for level in levels:
        if analysis_params.bw:
            if level > 0:
                linestyles.append('-')
            elif level == 0:
                linestyles.append((0, (5, 5)))
            else:
                linestyles.append((0, (1.0, 1.0)))
        else:
            # linestyles.append('-')
            if level > 0:
                linestyles.append('-')
            elif level == 0:
                linestyles.append((0, (5, 5)))
            else:
                linestyles.append('-')
    if np.any(LFP != np.zeros(LFP.shape)):
        im = ax.contour(X, Z, LFP,
                   levels=levels,
                   cmap='gray' if analysis_params.bw else 'RdBu',
                   vmin=-vlim,
                   vmax=vlim,
                   linewidths=3,
                   linestyles=linestyles,
                   zorder=-2,
                   rasterized=False)
        

        bbox = np.array(ax.get_position()).flatten()
        if cbar:
            cax = fig.add_axes((bbox[2]-0.01, 0.2, 0.01, 0.4), frameon=False)
            cbar = fig.colorbar(im, cax=cax, format=FormatStrFormatter('%i'), values=[-vlim, vlim])
            cbar.set_ticks(cbarticks)
            cbar.set_label('$\phi(\mathbf{r}, t)$ (nV)', labelpad=0)
            cbar.outline.set_visible(False)
        
        if show_legend:
            proxy = [plt.Line2D((0,1),(0,1), color='gray' if analysis_params.bw else plt.get_cmap('RdBu', 3)(2), ls='-', lw=3),
                     plt.Line2D((0,1),(0,1), color='gray' if analysis_params.bw else plt.get_cmap('RdBu', 3)(1), ls=(0, (5, 5)), lw=3),
                     plt.Line2D((0,1),(0,1), color='gray' if analysis_params.bw else plt.get_cmap('RdBu', 3)(0), ls=(0, (1, 1)), lw=3), ]
            
            ax.legend(proxy, [r'$\phi(\mathbf{r}, t) > 0$ nV',
                               r'$\phi(\mathbf{r}, t) = 0$ nV',
                               r'$\phi(\mathbf{r}, t) < 0$ nV'],
                      loc=1,
                      bbox_to_anchor=(1.2, 1),
                      fontsize=10,
                      frameon=False)
        
    zips = []
    for x, z in cell.get_idx_polygons():
        zips.append(zip(x, z))
    polycol = PolyCollection(zips,
                             edgecolors='k',
                             linewidths=0.5,
                             facecolors='k')
    ax.add_collection(polycol)
    
    ax.plot([100, 200], [-400, -400], 'k', lw=2, clip_on=False)
    ax.text(150, -470, r'100$\mu$m', va='center', ha='center')
    
    ax.axis('off')
    
    
    ax.plot(cell.xmid[cell.synidx],cell.zmid[cell.synidx], 'o', ms=6,
            markeredgecolor='k',
            markerfacecolor='w' if analysis_params.bw else 'r')
    
    color_vec = ['k' if analysis_params.bw else 'b', 'gray' if analysis_params.bw else 'g']
    for i in xrange(2):
        ax.plot(point_electrode_parameters['x'][i],
                        point_electrode_parameters['z'][i],'o',ms=6,
                        markeredgecolor='k',
                        markerfacecolor=color_vec[i])
    
    
    bbox = np.array(ax.get_position()).flatten()
    ax1 = fig.add_axes((bbox[0], bbox[1], 0.05, 0.2))
    ax1.plot(cell.tvec,point_electrode.LFP[0]*1e6,color=color_vec[0], clip_on=False)
    ax1.plot(cell.tvec,point_electrode.LFP[1]*1e6,color=color_vec[1], clip_on=False)
    axis = ax1.axis(ax1.axis('tight'))
    ax1.yaxis.set_major_locator(MaxNLocator(4))
    ax1.vlines(cell.tvec[tstep], axis[2], axis[3], lw=0.2)
    ax1.set_ylabel(r'$\phi(\mathbf{r}, t)$ (nV)', labelpad=0) #rotation='horizontal')
    ax1.set_xlabel('$t$ (ms)', labelpad=0)
    for loc, spine in ax1.spines.iteritems():
        if loc in ['right', 'top']:
            spine.set_color('none')            
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    
    ax2 = fig.add_axes((bbox[0], bbox[1]+.6, 0.05, 0.2))
    ax2.plot(cell.tvec,synapse.i*1E3, color='k' if analysis_params.bw else 'r',
             clip_on=False)
    axis = ax2.axis(ax2.axis('tight'))
    ax2.yaxis.set_major_locator(MaxNLocator(4))
    ax2.vlines(cell.tvec[tstep], axis[2], axis[3])
    ax2.set_ylabel(r'$I_{i, j}(t)$ (pA)', labelpad=0) #, rotation='horizontal')
    for loc, spine in ax2.spines.iteritems():
        if loc in ['right', 'top']:
            spine.set_color('none')            
    ax2.xaxis.set_ticks_position('bottom')
    ax2.yaxis.set_ticks_position('left')
    ax2.set_xticklabels([])
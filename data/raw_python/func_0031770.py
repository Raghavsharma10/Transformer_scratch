def quantity_yXL(fig, left, bottom, top, quantity=params.L_yXL, label=r'$\mathcal{L}_{yXL}$'):
                            
    '''make a bunch of image plots, each showing the spatial normalized
    connectivity of synapses'''
    
 
    layers = ['L1', 'L2/3', 'L4', 'L5', 'L6']
    ncols = len(params.y) / 4
    
    #assess vlims
    vmin = 0
    vmax = 0
    for y in params.y:
        if quantity[y].max() > vmax:
            vmax = quantity[y].max()
    
    gs = gridspec.GridSpec(4, 4, left=left, bottom=bottom, top=top)
    
    for i, y in enumerate(params.y):
        ax = fig.add_subplot(gs[i/4, i%4])

        masked_array = np.ma.array(quantity[y], mask=quantity[y]==0)
        # cmap = plt.get_cmap('hot', 20)
        # cmap.set_bad('k', 0.5)
        
        # im = ax.imshow(masked_array,
        im = ax.pcolormesh(masked_array,
                            vmin=vmin, vmax=vmax,
                            cmap=cmap,
                            #interpolation='nearest',
                            )
        ax.invert_yaxis()

        ax.axis(ax.axis('tight'))
        ax.xaxis.set_ticks_position('top')
        ax.set_xticks(np.arange(9)+0.5)
        ax.set_yticks(np.arange(5)+0.5)
        
        #if divmod(i, 4)[1] == 0:
        if i % 4 == 0:
            ax.set_yticklabels(layers, )
            ax.set_ylabel('$L$', labelpad=0.)
        else:
            ax.set_yticklabels([])
        if i < 4:
            ax.set_xlabel(r'$X$', labelpad=-1,fontsize=8)
            ax.set_xticklabels(params.X, rotation=270)
        else:
            ax.set_xticklabels([])
        ax.xaxis.set_label_position('top')
        
        ax.text(0.5, -0.13, r'$y=$'+y,
            horizontalalignment='center',
            verticalalignment='center',
            #
                transform=ax.transAxes,fontsize=5.5)
    
    #colorbar
    rect = np.array(ax.get_position().bounds)
    rect[0] += rect[2] + 0.01
    rect[1] = bottom
    rect[2] = 0.01
    rect[3] = top-bottom
    cax = fig.add_axes(rect)
    cbar = plt.colorbar(im, cax=cax)
    #cbar.set_label(label, ha='center')
    cbar.set_label(label, labelpad=0)
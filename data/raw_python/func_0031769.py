def cell_type_specificity(ax):
    '''make an imshow of the intranetwork connectivity'''
    masked_array = np.ma.array(params.T_yX, mask=params.T_yX==0)
    # cmap = plt.get_cmap('hot', 20)
    # cmap.set_bad('k', 0.5)
    # im = ax.imshow(masked_array, cmap=cmap, vmin=0, interpolation='nearest')
    im = ax.pcolormesh(masked_array, cmap=cmap, vmin=0, ) #interpolation='nearest')
    ax.axis(ax.axis('tight'))
    ax.invert_yaxis()
    ax.xaxis.set_ticks_position('top')
    ax.set_xticks(np.arange(9)+0.5)
    ax.set_yticks(np.arange(16)+0.5)
    ax.set_xticklabels(params.X, rotation=270)
    ax.set_yticklabels(params.y, )
    ax.xaxis.set_label_position('top')
    ax.set_xlabel(r'$X$', labelpad=-1,fontsize=8)
    ax.set_ylabel(r'$y$', labelpad=0, rotation=0,fontsize=8)

    rect = np.array(ax.get_position().bounds)
    rect[0] += rect[2] + 0.01
    rect[2] = 0.01
    fig = plt.gcf()
    cax = fig.add_axes(rect)

    cbar = plt.colorbar(im, cax=cax)
    #cbar.set_label(r'$\mathcal{T}_{yX}$', ha='center')
    cbar.set_label(r'$\mathcal{T}_{yX}$', labelpad=0)
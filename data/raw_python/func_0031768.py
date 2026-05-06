def connectivity(ax):

    '''make an imshow of the intranetwork connectivity'''
    masked_array = np.ma.array(params.C_YX, mask=params.C_YX==0)
    # if analysis_params.bw:
    #     cmap = plt.get_cmap(gray, 20)
    #     cmap.set_bad('k', 1.)
    # else:
    #     cmap = plt.get_cmap('hot', 20)
    #     cmap.set_bad('k', 0.5)
    # im = ax.imshow(masked_array, cmap=cmap, vmin=0, interpolation='nearest')
    im = ax.pcolormesh(masked_array, cmap=cmap, vmin=0, ) #interpolation='nearest')
    ax.axis(ax.axis('tight'))
    ax.invert_yaxis()
    ax.xaxis.set_ticks_position('top')
    ax.set_xticks(np.arange(9)+0.5)
    ax.set_yticks(np.arange(8)+0.5)
    ax.set_xticklabels(params.X, rotation=270)
    ax.set_yticklabels(params.Y, )
    ax.xaxis.set_label_position('top')
    ax.set_xlabel(r'$X$', labelpad=-1,fontsize=8)
    ax.set_ylabel(r'$Y$', labelpad=0, rotation=0,fontsize=8)

    rect = np.array(ax.get_position().bounds)
    rect[0] += rect[2] + 0.01
    rect[2] = 0.01
    fig = plt.gcf()
    cax = fig.add_axes(rect)

    cbar = plt.colorbar(im, cax=cax)
    #cbar.set_label(r'$C_{YX}$', ha='center')
    cbar.set_label(r'$C_{YX}$', labelpad=0)
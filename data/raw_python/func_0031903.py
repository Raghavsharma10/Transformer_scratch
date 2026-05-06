def plotConnectivity(ax):
    '''make an imshow of the intranetwork connectivity'''
    im = ax.pcolor(params.C_YX, cmap='hot')
    ax.axis(ax.axis('tight'))
    ax.invert_yaxis()
    ax.xaxis.set_ticks_position('top')
    ax.set_xticks(np.arange(9)+0.5)
    ax.set_yticks(np.arange(8)+0.5)
    ax.set_xticklabels(params.X, rotation=270)
    ax.set_yticklabels(params.Y, )
    #ax.set_ylabel(r'to ($Y$)', ha='center')
    #ax.set_xlabel(r'from ($X$)', va='center')
    ax.xaxis.set_label_position('top')

    rect = np.array(ax.get_position().bounds)
    rect[0] += rect[2] + 0.01
    rect[2] = 0.01
    fig = plt.gcf()
    cax = fig.add_axes(rect)

    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('connectivity', ha='center')
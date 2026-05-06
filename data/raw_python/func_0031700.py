def empty_bar_plot(ax):
    ''' Delete all axis ticks and labels '''
    plt.sca(ax)
    plt.setp(plt.gca(),xticks=[],xticklabels=[]) 
    return ax
def add_to_bar_plot(ax, x, number, name = '', color = '0.'):
    ''' This function takes an axes and adds one bar to it '''
    plt.sca(ax)
    plt.setp(ax,xticks=np.append(ax.get_xticks(),np.array([x]))\
             ,xticklabels=[item.get_text() for item in ax.get_xticklabels()] +[name])
    plt.bar([x],number , color = color, width = 1.)
    return ax
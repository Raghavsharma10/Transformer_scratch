def add_to_line_plot(ax, x, y, color = '0.' , label = ''):
    ''' This function takes an axes and adds one line to it '''
    plt.sca(ax)
    plt.plot(x,y, color = color, label = label)
    return ax
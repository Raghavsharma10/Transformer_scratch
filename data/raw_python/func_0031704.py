def frontiers_style():
    '''
    Figure styles for frontiers
    '''
    
    inchpercm = 2.54
    frontierswidth=8.5 
    textsize = 5
    titlesize = 7
    plt.rcdefaults()
    plt.rcParams.update({
        'figure.figsize' : [frontierswidth/inchpercm, frontierswidth/inchpercm],
        'figure.dpi' : 160,
        'xtick.labelsize' : textsize,
        'ytick.labelsize' : textsize,
        'font.size' : textsize,
        'axes.labelsize' : textsize,
        'axes.titlesize' : titlesize,
        'axes.linewidth': 0.75,
        'lines.linewidth': 0.75,
        'legend.fontsize' : textsize,
    })
    return None
def remove_axis_junk(ax, which=['right', 'top']):
    '''remove upper and right axis'''
    for loc, spine in ax.spines.items():
        if loc in which:
            spine.set_color('none')            
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
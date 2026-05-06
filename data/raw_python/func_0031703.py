def colorbar(fig, ax, im,
                 width=0.05,
                 height=1.0,
                 hoffset=0.01,
                 voffset=0.0,
                 orientation='vertical'):
    '''
    draw colorbar without resizing the axes object to make room

    kwargs:
    ::
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.AxesSubplot
        im : matplotlib.image.AxesImage
        width : float, colorbar width in fraction of ax width
        height : float, colorbar height in fraction of ax height
        hoffset : float, horizontal spacing to main axes in fraction of width
        voffset : float, vertical spacing to main axis in fraction of height
        orientation : str, 'horizontal' or 'vertical'

    return:
    ::
        object : colorbar handle

    '''
    rect = np.array(ax.get_position().bounds)
    
    rect = np.array(ax.get_position().bounds)
    caxrect = [0]*4
    caxrect[0] = rect[0] + rect[2] + hoffset*rect[2]
    caxrect[1] = rect[1] + voffset*rect[3]
    caxrect[2] = rect[2]*width
    caxrect[3] = rect[3]*height
    
    cax = fig.add_axes(caxrect)
    cb = fig.colorbar(im, cax=cax, orientation=orientation)
    
    return cb
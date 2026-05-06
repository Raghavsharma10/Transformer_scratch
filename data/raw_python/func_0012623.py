def relative_bias(fm,  scale_factor = 1, estimator = None):
    """
    Computes the relative bias, i.e. the distribution of saccade angles 
    and amplitudes. 

    Parameters:
        fm : DataMat
            The fixation data to use
        scale_factor : double
    Returns:
        2D probability distribution of saccade angles and amplitudes.
    """
    assert 'fix' in fm.fieldnames(), "Can not work without fixation  numbers"
    excl = fm.fix - np.roll(fm.fix, 1) != 1

    # Now calculate the direction where the NEXT fixation goes to
    diff_x = (np.roll(fm.x, 1) - fm.x)[~excl]
    diff_y = (np.roll(fm.y, 1) - fm.y)[~excl]
       

    # Make a histogram of diff values
    # this specifies left edges of the histogram bins, i.e. fixations between
    # ]0 binedge[0]] are included. --> fixations are ceiled
    ylim =  np.round(scale_factor * fm.image_size[0])
    xlim =  np.round(scale_factor * fm.image_size[1])
    x_steps = np.ceil(2*xlim) +1
    if x_steps % 2 != 0: x_steps+=1
    y_steps = np.ceil(2*ylim)+1
    if y_steps % 2 != 0: y_steps+=1
    e_x = np.linspace(-xlim,xlim,x_steps)
    e_y = np.linspace(-ylim,ylim,y_steps)

    #e_y = np.arange(-ylim, ylim+1)
    #e_x = np.arange(-xlim, xlim+1)
    samples = np.array(list(zip((scale_factor * diff_y),
                             (scale_factor* diff_x))))
    if estimator == None:
        (hist, _) = np.histogramdd(samples, (e_y, e_x))
    else:
        hist = estimator(samples, e_y, e_x)
    return hist
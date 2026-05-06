def slice_plot(*args, **kwargs):
    """Constructs a plot that lets you look at slices through a multidimensional array.
    
    Parameters
    ----------
    vals : array, (`M`, `D`, `P`, ...)
        Multidimensional array to visualize.
    x_vals_1 : array, (`M`,)
        Values along the first dimension.
    x_vals_2 : array, (`D`,)
        Values along the second dimension.
    x_vals_3 : array, (`P`,)
        Values along the third dimension.
        
        **...and so on. At least four arguments must be provided.**
    
    names : list of strings, optional
        Names for each of the parameters at hand. If None, sequential numerical
        identifiers will be used. Length must be equal to the number of
        dimensions of `vals`. Default is None.
    n : Positive int, optional
        Number of contours to plot. Default is 100.
    
    Returns
    -------
        f : :py:class:`Figure`
            The Matplotlib figure instance created.
    
    Raises
    ------
        GPArgumentError
            If the number of arguments is less than 4.
    """
    names = kwargs.get('names', None)
    n = kwargs.get('n', 100)
    num_axes = len(args) - 1
    if num_axes < 3:
        raise GPArgumentError("Must pass at least four arguments to slice_plot!")
    if num_axes != args[0].ndim:
        raise GPArgumentError("Number of dimensions of the first argument "
                              "must match the number of additional arguments "
                              "provided!")
    if names is None:
        names = [str(k) for k in range(2, num_axes)]
    f = plt.figure()
    height_ratios = [8]
    height_ratios += (num_axes - 2) * [1]
    gs = mplgs.GridSpec(num_axes - 2 + 1, 2, height_ratios=height_ratios, width_ratios=[8, 1])
    
    a_main = f.add_subplot(gs[0, 0])
    a_cbar = f.add_subplot(gs[0, 1])
    a_sliders = []
    for idx in xrange(0, num_axes - 2):
        a_sliders.append(f.add_subplot(gs[idx+1, :]))
    
    title = f.suptitle("")
    
    def update(val):
        """Update the slice shown.
        """
        a_main.clear()
        a_cbar.clear()
        idxs = [int(slider.val) for slider in sliders]
        vals = [args[k + 3][idxs[k]] for k in range(0, num_axes - 2)]
        descriptions = tuple(itertools.chain.from_iterable(itertools.izip(names[2:], vals)))
        fmt = "Slice" + (num_axes - 2) * ", %s: %f"
        title.set_text(fmt % descriptions)
        
        a_main.set_xlabel(names[1])
        a_main.set_ylabel(names[0])
        cs = a_main.contour(
            args[2],
            args[1],
            args[0][scipy.s_[:, :] + tuple(idxs)].squeeze(),
            n,
            vmin=args[0].min(),
            vmax=args[1].max()
        )
        cbar = f.colorbar(cs, cax=a_cbar)
        cbar.set_label("LL")
        
        f.canvas.draw()
    
    idxs_0 = (num_axes - 2) * [0]
    sliders = []
    for idx in xrange(0, num_axes - 2):
        sliders.append(
            mplw.Slider(
                a_sliders[idx],
                '%s index' % names[idx + 2],
                0,
                len(args[idx + 3]) - 1,
                valinit=idxs_0[idx],
                valfmt='%d'
            )
        )
        sliders[-1].on_changed(update)
    
    update(idxs_0)
    
    f.canvas.mpl_connect('key_press_event', lambda evt: arrow_respond(sliders[0], evt))
    
    return f
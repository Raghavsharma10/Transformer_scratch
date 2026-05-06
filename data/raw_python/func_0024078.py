def compute_stats(vals, check_nan=False, robust=False, axis=1, plot_QQ=False, bins=15, name=''):
    """Compute the average statistics (mean, std dev) for the given values.
    
    Parameters
    ----------
    vals : array-like, (`M`, `D`)
        Values to compute the average statistics along the specified axis of.
    check_nan : bool, optional
        Whether or not to check for (and exclude) NaN's. Default is False (do
        not attempt to handle NaN's).
    robust : bool, optional
        Whether or not to use robust estimators (median for mean, IQR for
        standard deviation). Default is False (use non-robust estimators).
    axis : int, optional
        Axis to compute the statistics along. Presently only supported if
        `robust` is False. Default is 1.
    plot_QQ : bool, optional
        Whether or not a QQ plot and histogram should be drawn for each channel.
        Default is False (do not draw QQ plots).
    bins : int, optional
        Number of bins to use when plotting histogram (for plot_QQ=True).
        Default is 15
    name : str, optional
        Name to put in the title of the QQ/histogram plot.
    
    Returns
    -------
    mean : ndarray, (`M`,)
        Estimator for the mean of `vals`.
    std : ndarray, (`M`,)
        Estimator for the standard deviation of `vals`.
    
    Raises
    ------
    NotImplementedError
        If `axis` != 1 when `robust` is True.
    NotImplementedError
        If `plot_QQ` is True.
    """
    if axis != 1 and robust:
        raise NotImplementedError("Values of axis other than 1 are not supported "
                                  "with the robust keyword at this time!")
    if robust:
        # TODO: This stuff should really be vectorized if there is something that allows it!
        if check_nan:
            mean = scipy.stats.nanmedian(vals, axis=axis)
            # TODO: HANDLE AXIS PROPERLY!
            std = scipy.zeros(vals.shape[0], dtype=float)
            for k in xrange(0, len(vals)):
                ch = vals[k]
                ok_idxs = ~scipy.isnan(ch)
                if ok_idxs.any():
                    std[k] = (scipy.stats.scoreatpercentile(ch[ok_idxs], 75) -
                              scipy.stats.scoreatpercentile(ch[ok_idxs], 25))
                else:
                    # Leave a nan where there are no non-nan values:
                    std[k] = scipy.nan
            std /= IQR_TO_STD
        else:
            mean = scipy.median(vals, axis=axis)
            # TODO: HANDLE AXIS PROPERLY!
            std = scipy.asarray([scipy.stats.scoreatpercentile(ch, 75.0) -
                                 scipy.stats.scoreatpercentile(ch, 25.0)
                                 for ch in vals]) / IQR_TO_STD
    else:
        if check_nan:
            mean = scipy.stats.nanmean(vals, axis=axis)
            std = scipy.stats.nanstd(vals, axis=axis)
        else:
            mean = scipy.mean(vals, axis=axis)
            std = scipy.std(vals, axis=axis)
    if plot_QQ:
        f = plt.figure()
        gs = mplgs.GridSpec(2, 2, height_ratios=[8, 1])
        a_QQ = f.add_subplot(gs[0, 0])
        a_hist = f.add_subplot(gs[0, 1])
        a_slider = f.add_subplot(gs[1, :])
        
        title = f.suptitle("")
        
        def update(val):
            """Update the index from the results to be displayed.
            """
            a_QQ.clear()
            a_hist.clear()
            idx = slider.val
            title.set_text("%s, n=%d" % (name, idx))
            
            nan_idxs = scipy.isnan(vals[idx, :])
            if not nan_idxs.all():
                osm, osr = scipy.stats.probplot(vals[idx, ~nan_idxs], dist='norm', plot=None, fit=False)
                a_QQ.plot(osm, osr, 'bo', markersize=10)
                a_QQ.set_title('QQ plot')
                a_QQ.set_xlabel('quantiles of $\mathcal{N}(0,1)$')
                a_QQ.set_ylabel('quantiles of data')
                
                a_hist.hist(vals[idx, ~nan_idxs], bins=bins, normed=True)
                locs = scipy.linspace(vals[idx, ~nan_idxs].min(), vals[idx, ~nan_idxs].max())
                a_hist.plot(locs, scipy.stats.norm.pdf(locs, loc=mean[idx], scale=std[idx]))
                a_hist.set_title('Normalized histogram and reported PDF')
                a_hist.set_xlabel('value')
                a_hist.set_ylabel('density')
            
            f.canvas.draw()
        
        def arrow_respond(slider, event):
            """Event handler for arrow key events in plot windows.

            Pass the slider object to update as a masked argument using a lambda function::

                lambda evt: arrow_respond(my_slider, evt)

            Parameters
            ----------
            slider : Slider instance associated with this handler.
            event : Event to be handled.
            """
            if event.key == 'right':
                slider.set_val(min(slider.val + 1, slider.valmax))
            elif event.key == 'left':
                slider.set_val(max(slider.val - 1, slider.valmin))

        slider = mplw.Slider(a_slider,
                             'index',
                             0,
                             len(vals) - 1,
                             valinit=0,
                             valfmt='%d')
        slider.on_changed(update)
        update(0)
        f.canvas.mpl_connect('key_press_event', lambda evt: arrow_respond(slider, evt))
    
    return (mean, std)
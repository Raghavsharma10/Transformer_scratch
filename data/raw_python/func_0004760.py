def residuals(cannon_set, dataset):
    """ Stack spectrum fit residuals, sort by each label. Include histogram of
    the RMS at each pixel.

    Parameters
    ----------
    cannon_set: Dataset
        best-fit Cannon spectra

    dataset: Dataset
        original spectra
    """
    print("Stacking spectrum fit residuals")
    res = dataset.test_fluxes - cannon_set.test_fluxes
    bad = dataset.test_ivars == SMALL**2
    err = np.zeros(len(dataset.test_ivars))
    err = np.sqrt(1. / dataset.test_ivars + 1. / cannon_set.test_ivars)
    res_norm = res / err
    res_norm = np.ma.array(res_norm,
                           mask=(np.ones_like(res_norm) *
                                 (np.std(res_norm,axis=0) == 0)))
    res_norm = np.ma.compress_cols(res_norm)

    for i in range(len(cannon_set.get_plotting_labels())):
        label_name = cannon_set.get_plotting_labels()[i]
        print("Plotting residuals sorted by %s" % label_name)
        label_vals = cannon_set.tr_label_vals[:,i]
        sorted_res = res_norm[np.argsort(label_vals)]
        mu = np.mean(sorted_res.flatten())
        sigma = np.std(sorted_res.flatten())
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        bottom_h = left_h = left+width+0.1
        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom_h, width, 0.1]
        rect_histy = [left_h, bottom, 0.1, height]
        plt.figure()
        axScatter = plt.axes(rect_scatter)
        axHistx = plt.axes(rect_histx)
        axHisty = plt.axes(rect_histy)
        im = axScatter.imshow(sorted_res, cmap=plt.cm.bwr_r,
                              interpolation="nearest", vmin=mu - 3. * sigma,
                              vmax=mu + 3. * sigma, aspect='auto',
                              origin='lower', extent=[0, len(dataset.wl),
                                                      min(label_vals),
                                                      max(label_vals)])
        cax, kw = colorbar.make_axes(axScatter.axes, location='bottom')
        plt.colorbar(im, cax=cax, orientation='horizontal')
        axScatter.set_title(
                r"Spectral Residuals Sorted by ${0:s}$".format(label_name))
        axScatter.set_xlabel("Pixels")
        axScatter.set_ylabel(r"$%s$" % label_name)
        axHisty.hist(np.std(res_norm,axis=1)[~np.isnan(np.std(res_norm, axis=1))], orientation='horizontal', range=[0,2])
        axHisty.axhline(y=1, c='k', linewidth=3, label="y=1")
        axHisty.legend(bbox_to_anchor=(0., 0.8, 1., .102),
                       prop={'family':'serif', 'size':'small'})
        axHisty.text(1.0, 0.5, "Distribution of Stdev of Star Residuals",
                     verticalalignment='center', transform=axHisty.transAxes,
                     rotation=270)
        axHisty.set_ylabel("Standard Deviation")
        start, end = axHisty.get_xlim()
        axHisty.xaxis.set_ticks(np.linspace(start, end, 3))
        axHisty.set_xlabel("Number of Stars")
        axHisty.xaxis.set_label_position("top")
        axHistx.hist(np.std(res_norm, axis=0)[~np.isnan(np.std(res_norm, axis=0))], range=[0.8,1.1])
        axHistx.axvline(x=1, c='k', linewidth=3, label="x=1")
        axHistx.set_title("Distribution of Stdev of Pixel Residuals")
        axHistx.set_xlabel("Standard Deviation")
        axHistx.set_ylabel("Number of Pixels")
        start, end = axHistx.get_ylim()
        axHistx.yaxis.set_ticks(np.linspace(start, end, 3))
        axHistx.legend()
        filename = "residuals_sorted_by_label_%s.png" % i
        plt.savefig(filename)
        print("File saved as %s" % filename)
        plt.close()

    # Auto-correlation of mean residuals
    print("Plotting Auto-Correlation of Mean Residuals")
    mean_res = res_norm.mean(axis=0)
    autocorr = np.correlate(mean_res, mean_res, mode="full")
    pkwidth = int(len(autocorr)/2-np.argmin(autocorr))
    xmin = int(len(autocorr)/2)-pkwidth
    xmax = int(len(autocorr)/2)+pkwidth
    zoom_x = np.linspace(xmin, xmax, len(autocorr[xmin:xmax]))
    fig, axarr = plt.subplots(2)
    axarr[0].plot(autocorr)
    axarr[0].set_title("Autocorrelation of Mean Spectral Residual")
    axarr[0].set_xlabel("Lag (# Pixels)")
    axarr[0].set_ylabel("Autocorrelation")
    axarr[1].plot(zoom_x, autocorr[xmin:xmax])
    axarr[1].set_title("Central Peak, Zoomed")
    axarr[1].set_xlabel("Lag (# Pixels)")
    axarr[1].set_ylabel("Autocorrelation")
    filename = "residuals_autocorr.png"
    plt.savefig(filename)
    print("saved %s" % filename)
    plt.close()
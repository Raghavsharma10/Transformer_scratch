def plot(ID, pipeline='everest2', show=True, campaign=None):
    '''
    Plots the de-trended flux for the given EPIC `ID` and for
    the specified `pipeline`.

    '''

    # Get the data
    time, flux = get(ID, pipeline=pipeline, campaign=campaign)

    # Remove nans
    mask = np.where(np.isnan(flux))[0]
    time = np.delete(time, mask)
    flux = np.delete(flux, mask)

    # Plot it
    fig, ax = pl.subplots(1, figsize=(10, 4))
    fig.subplots_adjust(bottom=0.15)
    ax.plot(time, flux, "k.", markersize=3, alpha=0.5)

    # Axis limits
    N = int(0.995 * len(flux))
    hi, lo = flux[np.argsort(flux)][[N, -N]]
    pad = (hi - lo) * 0.1
    ylim = (lo - pad, hi + pad)
    ax.set_ylim(ylim)

    # Show the CDPP
    from .k2 import CDPP
    ax.annotate('%.2f ppm' % CDPP(flux),
                xy=(0.98, 0.975), xycoords='axes fraction',
                ha='right', va='top', fontsize=12, color='r', zorder=99)

    # Appearance
    ax.margins(0, None)
    ax.set_xlabel("Time (BJD - 2454833)", fontsize=16)
    ax.set_ylabel("%s Flux" % pipeline.upper(), fontsize=16)
    fig.canvas.set_window_title("%s: EPIC %d" % (pipeline.upper(), ID))

    if show:
        pl.show()
        pl.close()
    else:
        return fig, ax
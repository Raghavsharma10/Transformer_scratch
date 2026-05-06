def overlay_histogram(df, path, palette=None):
    """
    Use plotly to create an overlay of length histograms
    Return html code, but also save as png

    Only has 10 colors, which get recycled up to 5 times.
    """
    if palette is None:
        palette = plotly.colors.DEFAULT_PLOTLY_COLORS * 5

    hist = Plot(path=path + "NanoComp_OverlayHistogram.html",
                title="Histogram of read lengths")
    hist.html, hist.fig = plot_overlay_histogram(df, palette, title=hist.title)
    hist.save()

    hist_norm = Plot(path=path + "NanoComp_OverlayHistogram_Normalized.html",
                     title="Normalized histogram of read lengths")
    hist_norm.html, hist_norm.fig = plot_overlay_histogram(
        df, palette, title=hist_norm.title, histnorm="probability")
    hist_norm.save()

    log_hist = Plot(path=path + "NanoComp_OverlayLogHistogram.html",
                    title="Histogram of log transformed read lengths")
    log_hist.html, log_hist.fig = plot_log_histogram(df, palette, title=log_hist.title)
    log_hist.save()

    log_hist_norm = Plot(path=path + "NanoComp_OverlayLogHistogram_Normalized.html",
                         title="Normalized histogram of log transformed read lengths")
    log_hist_norm.html, log_hist_norm.fig = plot_log_histogram(
        df, palette, title=log_hist_norm.title, histnorm="probability")
    log_hist_norm.save()

    return [hist, hist_norm, log_hist, log_hist_norm]
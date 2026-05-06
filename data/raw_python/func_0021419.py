def plot_log_histogram(df, palette, title, histnorm=""):
    """
    Plot overlaying histograms with log transformation of length
    Return both html and fig for png
    """
    data = [go.Histogram(x=np.log10(df.loc[df["dataset"] == d, "lengths"]),
                         opacity=0.4,
                         name=d,
                         histnorm=histnorm,
                         marker=dict(color=c))
            for d, c in zip(df["dataset"].unique(), palette)]
    xtickvals = [10**i for i in range(10) if not 10**i > 10 * np.amax(df["lengths"])]
    html = plotly.offline.plot(
        {"data": data,
         "layout": go.Layout(barmode='overlay',
                             title=title,
                             xaxis=dict(tickvals=np.log10(xtickvals),
                                        ticktext=xtickvals))},
        output_type="div",
        show_link=False)
    fig = go.Figure(
        {"data": data,
         "layout": go.Layout(barmode='overlay',
                             title=title,
                             xaxis=dict(tickvals=np.log10(xtickvals),
                                        ticktext=xtickvals))})
    return html, fig
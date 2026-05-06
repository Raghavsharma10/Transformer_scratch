def rug(x, label=None, opacity=None):
    """Rug chart.

    Parameters
    ----------
    x : array-like, optional
    label : TODO, optional
    opacity : TODO, optional

    Returns
    -------
    Chart

    """
    x = _try_pydatetime(x)
    x = np.atleast_1d(x)
    data = [
        go.Scatter(
            x=x,
            y=np.ones_like(x),
            name=label,
            opacity=opacity,
            mode='markers',
            marker=dict(symbol='line-ns-open'),
        )
    ]
    layout = dict(
        barmode='overlay',
        hovermode='closest',
        legend=dict(traceorder='reversed'),
        xaxis1=dict(zeroline=False),
        yaxis1=dict(
            domain=[0.85, 1],
            showline=False,
            showgrid=False,
            zeroline=False,
            anchor='free',
            position=0.0,
            showticklabels=False,
        ),
    )
    return Chart(data=data, layout=layout)
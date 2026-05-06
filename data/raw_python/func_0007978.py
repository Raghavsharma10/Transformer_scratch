def embed(contents='', width='100%', height=512, *args, **kwargs):
    """
    Embed geojson.io in an iframe in Jupyter/IPython notebook.

    Parameters
    ----------
    contents - see make_url()
    width - string, default '100%' - width of the iframe
    height - string / int, default 512 - height of the iframe
    kwargs - additional arguments are passed to `make_url()`

    """
    from IPython.display import HTML

    url = make_url(contents, *args, **kwargs)
    html = '<iframe src={url} width={width} height={height}></iframe>'.format(
        url=url, width=width, height=height)
    return HTML(html)
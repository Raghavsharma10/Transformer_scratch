def display_iframe_url(target, **kwargs):
    """Display the contents of a URL in an IPython notebook.
    
    :param target: the target url.
    :type target: string

    .. seealso:: `iframe_url()` for additional arguments."""

    txt = iframe_url(target, **kwargs)
    display(HTML(txt))
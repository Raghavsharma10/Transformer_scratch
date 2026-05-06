def display_url(target):
    """Displaying URL in an IPython notebook to allow the user to click and check on information. With thanks to Fernando Perez for putting together the implementation!
    :param target: the url to display.
    :type target: string."""

    prefix = u"http://" if not target.startswith("http") else u""
    target = prefix + target
    display(HTML(u'<a href="{t}" target=_blank>{t}</a>'.format(t=target)))
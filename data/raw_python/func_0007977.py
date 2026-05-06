def display(contents, domain=DEFAULT_DOMAIN, force_gist=False):
    """
    Open a web browser pointing to geojson.io with the specified content.

    If the content is large, an anonymous gist will be created on github and
    the URL will instruct geojson.io to download the gist data and then
    display. If the content is small, this step is not needed as the data can
    be included in the URL

    Parameters
    ----------
    content - (see make_geojson)
    domain - string, default http://geojson.io
    force_gist - bool, default False
        Create an anonymous gist on Github regardless of the size of the
        contents

    """
    url = make_url(contents, domain, force_gist)
    webbrowser.open(url)
    return url
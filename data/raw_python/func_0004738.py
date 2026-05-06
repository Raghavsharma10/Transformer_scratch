def render_image(**kwargs):
    """
    Unstrict template block for rendering an image:
    <img alt="{alt_text}" title="{title}" src="{url}">
    """
    html = ''

    url = kwargs.get('url', None)
    if url:
        html = '<img'

        alt_text = kwargs.get('alt_text', None)
        if alt_text:
            html += ' alt="{}"'.format(alt_text)

        title = kwargs.get('title', None)
        if title:
            html += ' title="{}"'.format(title)

        html += ' src="{}">'.format(url)

    return html
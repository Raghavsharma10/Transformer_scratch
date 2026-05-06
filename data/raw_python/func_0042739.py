def parse_tibiacom_content(content, *, html_class="BoxContent", tag="div", builder="lxml"):
    """Parses HTML content from Tibia.com into a BeautifulSoup object.

    Parameters
    ----------
    content: :class:`str`
        The raw HTML content from Tibia.com
    html_class: :class:`str`
        The HTML class of the parsed element. The default value is ``BoxContent``.
    tag: :class:`str`
        The HTML tag select. The default value is ``div``.
    builder: :class:`str`
        The builder to use. The default value is ``lxml``.

    Returns
    -------
    :class:`bs4.BeautifulSoup`, optional
        The parsed content.
    """
    return bs4.BeautifulSoup(content.replace('ISO-8859-1', 'utf-8'), builder,
                             parse_only=bs4.SoupStrainer(tag, class_=html_class))
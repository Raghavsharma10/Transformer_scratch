def markdown(text, mode='', context='', raw=False):
    """Render an arbitrary markdown document.

    :param str text: (required), the text of the document to render
    :param str mode: (optional), 'markdown' or 'gfm'
    :param str context: (optional), only important when using mode 'gfm',
        this is the repository to use as the context for the rendering
    :param bool raw: (optional), renders a document like a README.md, no gfm,
        no context
    :returns: str -- HTML formatted text

    """
    return gh.markdown(text, mode, context, raw)
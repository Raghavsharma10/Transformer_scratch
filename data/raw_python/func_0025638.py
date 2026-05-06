def action(elem, doc):
    """ Apply combined mustache template to all strings in document.
    """
    if type(elem) == Str and doc.mhash is not None:
        elem.text = doc.mrenderer.render(elem.text, doc.mhash)
        return elem
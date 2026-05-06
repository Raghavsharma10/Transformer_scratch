def process_tag(node):
    """
    Recursively go through a tag's children, converting them, then
    convert the tag itself.

    """
    text = ''

    exceptions = ['table']

    for element in node.children:
        if isinstance(element, NavigableString):
            text += element
        elif not node.name in exceptions:
            text += process_tag(element)

    try:
        convert_fn = globals()["convert_%s" % node.name.lower()]
        text = convert_fn(node, text)

    except KeyError:
        pass

    return text
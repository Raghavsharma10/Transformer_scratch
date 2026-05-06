def all_text(element):
    """
    A method for extending lxml's functionality, this will find and concatenate
    all text data that exists one level immediately underneath the given
    element. Unlike etree.tostring(element, method='text'), this will not
    recursively walk the entire underlying tree. It merely combines the element
    text attribute with the tail attribute of each child.
    """
    if element.text is None:
        text = []
    else:
        text = [element.text]
    tails = [child.tail for child in element if child.tail is not None]
    tails = [tail.strip() for tail in tails if tail.strip()]
    return ' '.join(text + tails)
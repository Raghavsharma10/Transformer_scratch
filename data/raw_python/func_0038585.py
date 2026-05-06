def parse_children(parent):
    """Recursively parse child tags until match is found"""

    components = []
    for tag in parent.children:
        matched = parse_tag(tag)
        if matched:
            components.append(matched)
        elif hasattr(tag, 'contents'):
            components += parse_children(tag)
    return components
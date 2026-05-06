def node_contents_str(tag):
    """
    Return the contents of a tag, including it's children, as a string.
    Does not include the root/parent of the tag.
    """
    if not tag:
        return None
    tag_string = ''
    for child_tag in tag.children:
        if isinstance(child_tag, Comment):
            # BeautifulSoup does not preserve comment tags, add them back
            tag_string += '<!--%s-->' % unicode_value(child_tag)
        else:
            tag_string += unicode_value(child_tag)
    return tag_string if tag_string != '' else None
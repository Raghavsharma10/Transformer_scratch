def get_first_element_index(root, tag_name):
    """
    In order to use Element.insert() in a convenient way,
    this function will find the first child tag with tag_name
    and return its index position
    The index can then be used to insert an element before or after the
    found tag using Element.insert()
    """
    tag_index = 1
    for tag in root:
        if tag.tag == tag_name:
            # Return the first one found if there is a match
            return tag_index
        tag_index = tag_index + 1
    # Default
    return None
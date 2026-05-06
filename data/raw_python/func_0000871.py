def add_tag_before(tag_name, tag_text, parent_tag, before_tag_name):
    """
    Helper function to refactor the adding of new tags
    especially for when converting text to role tags
    """
    new_tag = Element(tag_name)
    new_tag.text = tag_text
    if get_first_element_index(parent_tag, before_tag_name):
        parent_tag.insert( get_first_element_index(parent_tag, before_tag_name) - 1, new_tag)
    return parent_tag
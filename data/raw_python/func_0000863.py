def tag_limit_sibling_ordinal(tag, stop_tag_name):
    """
    Count previous tags of the same name until it
    reaches a tag name of type stop_tag, then stop counting
    """
    tag_count = 1
    for prev_tag in tag.previous_elements:
        if prev_tag.name == tag.name:
            tag_count += 1
        if prev_tag.name == stop_tag_name:
            break

    return tag_count
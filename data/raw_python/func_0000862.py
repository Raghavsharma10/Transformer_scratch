def tag_fig_ordinal(tag):
    """
    Meant for finding the position of fig tags with respect to whether
    they are for a main figure or a child figure
    """
    tag_count = 0
    if 'specific-use' not in tag.attrs:
        # Look for tags with no "specific-use" attribute
        return len(list(filter(lambda tag: 'specific-use' not in tag.attrs,
                          tag.find_all_previous(tag.name)))) + 1
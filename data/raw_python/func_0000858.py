def component_acting_parent_tag(parent_tag, tag):
    """
    Only intended for use in getting components, look for tag name of fig-group
    and if so, find the first fig tag inside it as the acting parent tag
    """
    if parent_tag.name == "fig-group":
        if (len(tag.find_previous_siblings("fig")) > 0):
            acting_parent_tag = first(extract_nodes(parent_tag, "fig"))
        else:
            # Do not return the first fig as parent of itself
            return None
    else:
        acting_parent_tag = parent_tag
    return acting_parent_tag
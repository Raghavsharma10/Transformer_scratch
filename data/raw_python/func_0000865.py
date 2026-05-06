def tag_supplementary_material_sibling_ordinal(tag):
    """
    Strategy is to count the previous supplementary-material tags
    having the same asset value to get its sibling ordinal.
    The result is its position inside any parent tag that
    are the same asset type
    """
    if hasattr(tag, 'name') and tag.name != 'supplementary-material':
        return None

    nodenames = ['fig','media','sub-article']
    first_parent_tag = first_parent(tag, nodenames)

    sibling_ordinal = 1

    if first_parent_tag:
        # Within the parent tag of interest, count the tags
        #  having the same asset value
        for supp_tag in first_parent_tag.find_all(tag.name):
            if tag == supp_tag:
                # Stop once we reach the same tag we are checking
                break
            if supp_asset(supp_tag) == supp_asset(tag):
                sibling_ordinal += 1

    else:
        # Look in all previous elements that do not have a parent
        #  and count the tags having the same asset value
        for prev_tag in tag.find_all_previous(tag.name):
            if not first_parent(prev_tag, nodenames):
                if supp_asset(prev_tag) == supp_asset(tag):
                    sibling_ordinal += 1

    return sibling_ordinal
def contrib_inline_aff(contrib_tag):
    """
    Given a contrib tag, look for an aff tag directly inside it
    """
    aff_tags = []
    for child_tag in contrib_tag:
        if child_tag and child_tag.name and child_tag.name == "aff":
            aff_tags.append(child_tag)
    return aff_tags
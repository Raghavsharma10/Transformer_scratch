def title_tag_inspected(tag, parent_tag_name=None, p_parent_tag_name=None, direct_sibling_only=False):
    """Extract the title tag and sometimes inspect its parents"""

    title_tag = None
    if direct_sibling_only is True:
        for sibling_tag in tag:
            if sibling_tag.name and sibling_tag.name == "title":
                title_tag = sibling_tag
    else:
        title_tag = raw_parser.title(tag)

    if parent_tag_name and p_parent_tag_name:
        if (title_tag and title_tag.parent.name and title_tag.parent.parent.name
            and title_tag.parent.name == parent_tag_name
            and title_tag.parent.parent.name == p_parent_tag_name):
            pass
        else:
            title_tag = None

    return title_tag
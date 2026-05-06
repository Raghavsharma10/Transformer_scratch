def title_text(tag, parent_tag_name=None, p_parent_tag_name=None, direct_sibling_only=False):
    """Extract the text of a title tag and sometimes inspect its parents"""
    title = None

    title_tag = title_tag_inspected(tag, parent_tag_name, p_parent_tag_name, direct_sibling_only)

    if title_tag:
        title = node_contents_str(title_tag)
    return title
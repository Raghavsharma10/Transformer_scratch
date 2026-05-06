def diff(old_html, new_html, cutoff=0.0, plaintext=False, pretty=False):
    """Show the differences between the old and new html document, as html.

    Return the document html with extra tags added to show changes. Add <ins>
    tags around newly added sections, and <del> tags to show sections that have
    been deleted.
    """
    if plaintext:
        old_dom = parse_text(old_html)
        new_dom = parse_text(new_html)
    else:
        old_dom = parse_minidom(old_html)
        new_dom = parse_minidom(new_html)

    # If the two documents are not similar enough, don't show the changes.
    if not check_text_similarity(old_dom, new_dom, cutoff):
        return '<h2>The differences from the previous version are too large to show concisely.</h2>'

    dom = dom_diff(old_dom, new_dom)

    # HTML-specific cleanup.
    if not plaintext:
        fix_lists(dom)
        fix_tables(dom)

    # Only return html for the document body contents.
    body_elements = dom.getElementsByTagName('body')
    if len(body_elements) == 1:
        dom = body_elements[0]

    return minidom_tostring(dom, pretty=pretty)
def extract_text(html,
                 guess_punct_space=True,
                 guess_layout=True,
                 newline_tags=NEWLINE_TAGS,
                 double_newline_tags=DOUBLE_NEWLINE_TAGS):
    """
    Convert html to text, cleaning invisible content such as styles.

    Almost the same as normalize-space xpath, but this also
    adds spaces between inline elements (like <span>) which are
    often used as block elements in html markup, and adds appropriate
    newlines to make output better formatted.

    html should be a unicode string or an already parsed lxml.html element.

    ``html_text.etree_to_text`` is a lower-level function which only accepts
    an already parsed lxml.html Element, and is not doing html cleaning itself.

    When guess_punct_space is True (default), no extra whitespace is added
    for punctuation. This has a slight (around 10%) performance overhead
    and is just a heuristic.

    When guess_layout is True (default), a newline is added
    before and after ``newline_tags`` and two newlines are added before
    and after ``double_newline_tags``. This heuristic makes the extracted
    text more similar to how it is rendered in the browser.

    Default newline and double newline tags can be found in
    `html_text.NEWLINE_TAGS` and `html_text.DOUBLE_NEWLINE_TAGS`.
    """
    if html is None:
        return ''
    cleaned = _cleaned_html_tree(html)
    return etree_to_text(
        cleaned,
        guess_punct_space=guess_punct_space,
        guess_layout=guess_layout,
        newline_tags=newline_tags,
        double_newline_tags=double_newline_tags,
    )
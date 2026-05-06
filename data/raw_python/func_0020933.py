def strip_outer_tag(text):
    """Strips the outer tag, if text starts with a tag.  Not entity aware;
    designed to quickly strip outer tags from lxml cleaner output.  Only
    checks for <p> and <div> outer tags."""
    if not text or not isinstance(text, basestring):
        return text
    stripped = text.strip()
    if (stripped.startswith('<p>') or stripped.startswith('<div>')) and \
        (stripped.endswith('</p>') or stripped.endswith('</div>')):
        return stripped[stripped.index('>')+1:stripped.rindex('<')]
    return text
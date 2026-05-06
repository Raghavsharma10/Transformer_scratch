def _clean_text(text):
    """
    Clean up a multiple-line, potentially multiple-paragraph text
    string.  This is used to extract the first paragraph of a string
    and eliminate line breaks and indentation.  Lines will be joined
    together by a single space.

    :param text: The text string to clean up.  It is safe to pass
                 ``None``.

    :returns: The first paragraph, cleaned up as described above.
    """

    desc = []
    for line in (text or '').strip().split('\n'):
        # Clean up the line...
        line = line.strip()

        # We only want the first paragraph
        if not line:
            break

        desc.append(line)

    return ' '.join(desc)
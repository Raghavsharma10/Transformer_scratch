def remove_whitespace(text):
    """Remove unnecessary whitespace while keeping logical structure.

    Keyword arguments:
    text -- text to remove whitespace from (list)

    Retain paragraph structure but remove other whitespace,
    such as between words on a line and at the start and end of the text.
    """
    clean_text = []
    curr_line = ''
    # Remove any newlines that follow two lines of whitespace consecutively
    # Also remove whitespace at start and end of text
    while text:
        if not curr_line:
            # Find the first line that is not whitespace and add it
            curr_line = text.pop(0)
            while not curr_line.strip() and text:
                curr_line = text.pop(0)
            if curr_line.strip():
                clean_text.append(curr_line)
        else:
            # Filter the rest of the lines
            curr_line = text.pop(0)
            if not text:
                # Add the final line if it is not whitespace
                if curr_line.strip():
                    clean_text.append(curr_line)
                continue

            if curr_line.strip():
                clean_text.append(curr_line)
            else:
                # If the current line is whitespace then make sure there is
                # no more than one consecutive line of whitespace following
                if not text[0].strip():
                    if len(text) > 1 and text[1].strip():
                        clean_text.append(curr_line)
                else:
                    clean_text.append(curr_line)

    # Now filter each individual line for extraneous whitespace
    cleaner_text = []
    for line in clean_text:
        clean_line = ' '.join(line.split())
        if not clean_line.strip():
            clean_line += '\n'
        cleaner_text.append(clean_line)
    return cleaner_text
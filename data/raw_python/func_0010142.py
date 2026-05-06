def re_filter(text, regexps):
    """Filter text using regular expressions."""
    if not regexps:
        return text

    matched_text = []
    compiled_regexps = [re.compile(x) for x in regexps]
    for line in text:
        if line in matched_text:
            continue

        for regexp in compiled_regexps:
            found = regexp.search(line)
            if found and found.group():
                matched_text.append(line)

    return matched_text or text
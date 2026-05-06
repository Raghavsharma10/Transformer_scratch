def parse_comment(doc_comment, next_line):
    r"""
    Split the raw comment text into a dictionary of tags.  The main comment
    body is included as 'doc'.

    >>> comment = get_doc_comments(read_file('examples/module.js'))[4][0]
    >>> parse_comment(strip_stars(comment), '')['doc']
    'This is the documentation for the fourth function.\n\n Since the function being documented is itself generated from another\n function, its name needs to be specified explicitly. using the @function tag'
    >>> parse_comment(strip_stars(comment), '')['function']
    'not_auto_discovered'

    If there are multiple tags with the same name, they're included as a list:

    >>> parse_comment(strip_stars(comment), '')['param']
    ['{String} arg1 The first argument.', '{Int} arg2 The second argument.']

    """
    sections = re.split('\n\s*@', doc_comment)
    tags = { 
        'doc': sections[0].strip(),
        'guessed_function': guess_function_name(next_line),
        'guessed_params': guess_parameters(next_line)
    }
    for section in sections[1:]:
        tag, body = split_tag(section)
        if tag in tags:
            existing = tags[tag]
            try:
                existing.append(body)
            except AttributeError:
                tags[tag] = [existing, body]
        else:
            tags[tag] = body
    return tags
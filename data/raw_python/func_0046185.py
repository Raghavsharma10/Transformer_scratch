def guess_function_name(next_line, regexps=FUNCTION_REGEXPS):
    """
    Attempt to determine the function name from the first code line
    following the comment.  The patterns recognized are described by
    `regexps`, which defaults to FUNCTION_REGEXPS.  If a match is successful, 
    returns the function name.  Otherwise, returns None.
    """
    for regexp in regexps:
        match = re.search(regexp, next_line)
        if match:
            return match.group(1)
    return None
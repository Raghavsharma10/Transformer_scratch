def removeEscapes(value, quoted=0):

    """Remove escapes from in front of quotes (which IRAF seems to
    just stick in for fun sometimes.)  Remove \-newline too.
    If quoted is true, removes all blanks following \-newline
    (which is a nasty thing IRAF does for continuations inside
    quoted strings.)
    XXX Should we remove \\ too?
    """

    i = value.find(r'\"')
    while i>=0:
        value = value[:i] + value[i+1:]
        i = value.find(r'\"',i+1)
    i = value.find(r"\'")
    while i>=0:
        value = value[:i] + value[i+1:]
        i = value.find(r"\'",i+1)
    # delete backslash-newlines
    i = value.find("\\\n")
    while i>=0:
        j = i+2
        if quoted:
            # ignore blanks and tabs following \-newline in quoted strings
            for c in value[i+2:]:
                if c not in ' \t':
                    break
                j = j+1
        value = value[:i] + value[j:]
        i = value.find("\\\n",i+1)
    return value
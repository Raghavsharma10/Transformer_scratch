def translateName(s, dot=0):

    """Convert CL parameter or variable name to Python-acceptable name

    Translate embedded dollar signs to 'DOLLAR'
    Add 'PY' prefix to components that are Python reserved words
    Add 'PY' prefix to components start with a number
    If dot != 0, also replaces '.' with 'DOT'
    """

    s = s.replace('$', 'DOLLAR')
    sparts = s.split('.')
    for i in range(len(sparts)):
        if sparts[i] == "" or sparts[i][0] in string.digits or \
          keyword.iskeyword(sparts[i]):
            sparts[i] = 'PY' + sparts[i]
    if dot:
        return 'DOT'.join(sparts)
    else:
        return '.'.join(sparts)
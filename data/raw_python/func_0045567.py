def substitute(search, replace, text):
    'Regex substitution function. Replaces regex ``search`` with ``replace`` in ``text``'
    return re.sub(re.compile(str(search)), replace, text)
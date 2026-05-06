def search(pattern, text):
    'Regex pattern search. Returns match if ``pattern`` is found in ``text``'
    return re.compile(str(pattern)).search(str(text))
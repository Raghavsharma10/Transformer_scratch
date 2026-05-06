def init():
    '''
    Load in the Chinese-English dictionary. This takes 1-2 seconds. It
    is done when the other functions are used, but this is public since
    preloading sometimes makes sense.
    '''
    global dictionaries, trees

    dictionaries = {
        'traditional': {},
        'simplified': {}
    }

    trees = {
        'traditional': Tree(),
        'simplified': Tree()
    }

    lines = gzip.open(
        os.path.join(os.path.dirname(__file__), "cedict.txt.gz"),
        mode='rt',
        encoding='utf-8'
    )
    exp = re.compile("^([^ ]+) ([^ ]+) \[(.*)\] /(.+)/")
    parsed_lines = (exp.match(line).groups()
                    for line in lines
                    if line[0] != '#')

    for traditional, simplified, pinyin, meaning in parsed_lines:
        meaning = meaning.split('/')
        dictionaries['traditional'][traditional] = meaning
        dictionaries['simplified'][simplified] = meaning
        _add_to_tree(trees['traditional'], traditional, meaning)
        _add_to_tree(trees['simplified'], simplified, meaning)
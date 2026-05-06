def char_sets():
    """Return a list of the IANA Character Sets, or an empty list if the
    IANA website is unreachable.
    Store it as a function attribute so that we only build the list once.
    """
    if not hasattr(char_sets, 'setlist'):
        clist = []
        try:
            data = requests.get('http://www.iana.org/assignments/character-'
                                'sets/character-sets-1.csv')
        except requests.exceptions.RequestException:
            return []

        for line in data.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.count(',') > 0:
                    vals = line.split(',')
                    if vals[0]:
                        clist.append(vals[0])
                    else:
                        clist.append(vals[1])

        char_sets.setlist = clist
    return char_sets.setlist
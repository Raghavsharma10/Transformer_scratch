def media_types():
    """Return a list of the IANA Media (MIME) Types, or an empty list if the
    IANA website is unreachable.
    Store it as a function attribute so that we only build the list once.
    """
    if not hasattr(media_types, 'typelist'):
        tlist = []
        categories = [
            'application',
            'audio',
            'font',
            'image',
            'message',
            'model',
            'multipart',
            'text',
            'video'
        ]
        for cat in categories:
            try:
                data = requests.get('http://www.iana.org/assignments/'
                                    'media-types/%s.csv' % cat)
            except requests.exceptions.RequestException:
                return []

            types = []
            for line in data.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.count(',') > 0:
                        reg_template = line.split(',')[1]
                        if reg_template:
                            types.append(reg_template)
                        else:
                            types.append(cat + '/' + line.split(',')[0])

            tlist.extend(types)
        media_types.typelist = tlist
    return media_types.typelist
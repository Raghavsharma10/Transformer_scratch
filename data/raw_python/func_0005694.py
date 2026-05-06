def get_content(api, rebuild_cache=False):
    """get content from server or cache"""
    if hasattr(get_content, 'cache') and not rebuild_cache:
        return get_content.cache
    if not os.path.exists(CONTENT_JSON) or rebuild_cache:
        import locale
        content_endpoint = api.content.get
        # pylint: disable=protected-access
        try_langs = []
        try:
            lang = get_translation_for('habitipy').info()['language']
            try_langs.append(lang)
        except KeyError:
            pass
        try:
            loc = locale.getdefaultlocale()[0]
            if loc:
                try_langs.append(loc)
                try_langs.append(loc[:2])
        except IndexError:
            pass
        server_lang = content_endpoint._node.params['query']['language']
        # handle something like 'ru_RU' not available - only 'ru'
        for lang in try_langs:
            if lang in server_lang.possible_values:
                loc = {'language': lang}
                break
        else:
            loc = {}
        get_content.cache = content = content_endpoint(**loc)
        with open(CONTENT_JSON, 'w') as f:
            json.dump(content, f)
        return content
    try:
        with open(CONTENT_JSON) as f:
            get_content.cache = content = json.load(f)
        return content
    except JSONDecodeError:
        return get_content(api, rebuild_cache=True)
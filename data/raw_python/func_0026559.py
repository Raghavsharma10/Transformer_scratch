def tabset(titles, contents):
    """A tabbed container widget"""

    tabs = []
    for no, title in enumerate(titles):
        tab = {
            'title': title,
        }
        content = contents[no]
        if isinstance(content, list):
            tab['items'] = content
        else:
            tab['items'] = [content]
        tabs.append(tab)

    result = {
        'type': 'tabs',
        'tabs': tabs
    }

    return result
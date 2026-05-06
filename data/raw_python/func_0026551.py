def _augment_info(info):
    """Fill out the template information"""

    info['description_header'] = "=" * len(info['description'])
    info['component_name'] = info['plugin_name'].capitalize()
    info['year'] = time.localtime().tm_year
    info['license_longtext'] = ''

    info['keyword_list'] = u""
    for keyword in info['keywords'].split(" "):
        print(keyword)
        info['keyword_list'] += u"\'" + str(keyword) + u"\', "
    print(info['keyword_list'])
    if len(info['keyword_list']) > 0:
        # strip last comma
        info['keyword_list'] = info['keyword_list'][:-2]

    return info
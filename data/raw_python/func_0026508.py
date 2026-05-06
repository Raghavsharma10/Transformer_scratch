def format_template(template, content):
    """Render a given pystache template
    with given content"""

    import pystache
    result = u""
    if True:  # try:
        result = pystache.render(template, content, string_encoding='utf-8')
    # except (ValueError, KeyError) as e:
    #    print("Templating error: %s %s" % (e, type(e)))

    # pprint(result)
    return result
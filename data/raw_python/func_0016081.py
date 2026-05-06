def check_content(content, **kwargs):
    '''check content for "active" urls'''
    # valid html root tag
    try:
        # render elements tree from content
        tree = fragment_fromstring(content)
        # flag for prevent content rerendering, when no "active" urls found
        processed = False
        # django > 1.5 template boolean\None variables feature
        if isinstance(kwargs['parent_tag'], bool):
            if not kwargs['parent_tag']:
                kwargs['parent_tag'] = 'self'
            else:
                raise ImproperlyConfigured('''
                    parent_tag=True is not allowed
                ''')
        elif kwargs['parent_tag'] is None:
            kwargs['parent_tag'] = 'self'
        # if parent_tag is False\None\''\a\self
        # "active" status will be applied directly to "<a>"
        if kwargs['parent_tag'].lower() in ('a', 'self', ''):
            # xpath query to get all "<a>"
            urls = tree.xpath('.//a')
            # check "active" status for all urls
            for url in urls:
                if check_active(url, url, **kwargs):
                    # mark flag for rerendering content
                    processed = True
        # otherwise css_class must be applied to parent_tag
        else:
            # xpath query to get all parent tags
            elements = tree.xpath('.//{parent_tag}'.format(
                parent_tag=kwargs['parent_tag'],
            ))
            # check all elements for "active" "<a>"
            for element in elements:
                # xpath query to get all "<a>"
                urls = element.xpath('.//a')
                # check "active" status for all urls
                for url in urls:
                    if check_active(url, element, **kwargs):
                        # flag for rerendering content tree
                        processed = True
                        # stop checking other "<a>"
                        break
        # do not rerender content if no "active" urls found
        if processed:
            # render content from tree
            return tostring(tree, encoding='unicode')
    # not valid html root tag
    except ParserError:
        # raise an exception with configuration example
        raise ImproperlyConfigured('''
            content of {% activeurl %} must have valid html root tag
            for example
                {% activeurl %}
                    <ul>
                        <li>
                            <a href="/page/">page</a>
                        </li>
                        <li>
                            <a href="/other_page/">other_page</a>
                        </li>
                    </ul>
                {% endactiveurl %}
            in this case <ul> is valid content root tag
        ''')
    return content
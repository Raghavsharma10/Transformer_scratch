def check_active(url, element, **kwargs):
    '''check "active" url, apply css_class'''
    menu = yesno_to_bool(kwargs['menu'], 'menu')
    ignore_params = yesno_to_bool(kwargs['ignore_params'], 'ignore_params')

    # check missing href parameter
    if not url.attrib.get('href', None) is None:
        # get href attribute
        href = url.attrib['href'].strip()

        # href="#" is often used when links shouldn't be handled by browsers.
        # For example, Bootstrap uses this for expandable menus on
        # small screens, see
        # https://getbootstrap.com/docs/4.0/components/navs/#using-dropdowns
        if href == '#':
            return False

        # split into urlparse object
        href = urlparse.urlsplit(href)

        # cut off hashtag (anchor)
        href = href._replace(fragment='')

        # cut off get params (?key=var&etc=var2)
        if ignore_params:
            href = href._replace(query='')
            kwargs['full_path'] = urlparse.urlunsplit(
                urlparse.urlsplit(
                    kwargs['full_path']
                )._replace(query='')
            )

        # build urlparse object back into string
        href = urlparse.urlunsplit(href)

        # check empty href
        if href == '':
            # replace href with current location
            href = kwargs['full_path']
        # compare full_path with href according to menu configuration

        if menu:
            # try mark "root" (/) url as "active", in equals way
            if href == '/' == kwargs['full_path']:
                logic = True
            # skip "root" (/) url, otherwise it will be always "active"
            elif href != '/':
                # start with logic
                logic = (
                    kwargs['full_path'].startswith(href)
                    or
                    # maybe an urlquoted href was supplied
                    urlquote(kwargs['full_path']).startswith(href)
                    or
                    kwargs['full_path'].startswith(urlquote(href))
                )
            else:
                logic = False
        else:
            # equals logic
            logic = (
                kwargs['full_path'] == href
                or
                # maybe an urlquoted href was supplied
                urlquote(kwargs['full_path']) == href
                or
                kwargs['full_path'] == urlquote(href)
            )
        # "active" url found
        if logic:
            # check parent tag has "class" attribute or it is empty
            if element.attrib.get('class'):
                # prevent multiple "class" attribute adding
                if kwargs['css_class'] not in element.attrib['class']:
                    # append "active" class
                    element.attrib['class'] += ' {css_class}'.format(
                        css_class=kwargs['css_class'],
                    )
            else:
                # create or set (if empty) "class" attribute
                element.attrib['class'] = kwargs['css_class']
            return True
    # no "active" urls found
    return False
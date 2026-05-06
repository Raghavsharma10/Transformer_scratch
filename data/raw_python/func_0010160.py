def write_part_file(args, url, raw_html, html=None, part_num=None):
    """Write PART.html file(s) to disk, images in PART_files directory.

    Keyword arguments:
    args -- program arguments (dict)
    raw_html -- unparsed HTML file content (list)
    html -- parsed HTML file content (lxml.html.HtmlElement) (default: None)
    part_num -- PART(#).html file number (int) (default: None)
    """
    if part_num is None:
        part_num = get_num_part_files() + 1
    filename = 'PART{0}.html'.format(part_num)

    # Decode bytes to string in Python 3 versions
    if not PY2 and isinstance(raw_html, bytes):
        raw_html = raw_html.encode('ascii', 'ignore')

    # Convert html to an lh.HtmlElement object for parsing/saving images
    if html is None:
        html = lh.fromstring(raw_html)

    # Parse HTML if XPath entered
    if args['xpath']:
        raw_html = parse_html(html, args['xpath'])
        if isinstance(raw_html, list):
            if not isinstance(raw_html[0], lh.HtmlElement):
                raise ValueError('XPath should return an HtmlElement object.')
        else:
            if not isinstance(raw_html, lh.HtmlElement):
                raise ValueError('XPath should return an HtmlElement object.')

    # Write HTML and possibly images to disk
    if raw_html:
        if not args['no_images'] and (args['pdf'] or args['html']):
            raw_html = write_part_images(url, raw_html, html, filename)
        with open(filename, 'w') as part:
            if not isinstance(raw_html, list):
                raw_html = [raw_html]
                if isinstance(raw_html[0], lh.HtmlElement):
                    for elem in raw_html:
                        part.write(lh.tostring(elem))
                else:
                    for line in raw_html:
                        part.write(line)
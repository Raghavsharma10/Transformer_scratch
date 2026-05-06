def generate_page_object(page_name, url):
    "Generate page object from URL"

    # Attempt to extract partial URL for verification.
    url_with_path = u('^.*//[^/]+([^?]+)?|$')
    try:
        match = re.match(url_with_path, url)
        partial_url = match.group(1)
        print("Using partial URL for location verification. ", partial_url)
    except:
        # use full url since we couldn't extract a partial.
        partial_url = url
        print("Could not find usable partial url, using full url.", url)

    # Attempt to map input objects.
    print("Processing page source...")
    response = urllib2.urlopen(url)
    html = response.read()
    input_tags_expr = u('<\s*input[^>]*>')
    input_tag_iter = re.finditer(input_tags_expr, html, re.IGNORECASE)

    objectmap = ""
    print("Creating object map for <input> tags...")
    for input_tag_match in input_tag_iter:
        if not "hidden" in input_tag_match.group(0):
            try:
                print("processing", input_tag_match.group(0))
                obj_map_entry = _process_input_tag(input_tag_match.group(0))
                objectmap += u("    ") + obj_map_entry + "\n"
            except Exception as e:
                print(e)
                # we failed to process it, nothing more we can do.
                pass

    return _page_object_template_.contents.format(date=datetime.now(),
                                                  url=url,
                                                  pagename=page_name,
                                                  partialurl=partial_url,
                                                  objectmap=objectmap)
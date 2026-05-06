def self_uri(soup):
    """
    self-uri tags
    """

    self_uri = []
    self_uri_tags = raw_parser.self_uri(soup)
    position = 1
    for tag in self_uri_tags:
        item = {}

        copy_attribute(tag.attrs, 'xlink:href', item, 'xlink_href')
        copy_attribute(tag.attrs, 'content-type', item)

        # Get the tag type
        nodenames = ["sub-article"]
        details = tag_details(tag, nodenames)
        copy_attribute(details, 'type', item)

        # Increment the position
        item['position'] = position
        # Ordinal should be the same as position in this case but set it anyway
        item['ordinal'] = tag_ordinal(tag)

        self_uri.append(item)

    return self_uri
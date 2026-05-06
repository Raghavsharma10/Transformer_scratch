def inline_graphics(soup):
    """
    inline-graphic tags
    """
    inline_graphics = []

    inline_graphic_tags = raw_parser.inline_graphic(soup)

    position = 1

    for tag in inline_graphic_tags:
        item = {}

        copy_attribute(tag.attrs, 'xlink:href', item, 'xlink_href')

        # Get the tag type
        nodenames = ["sub-article"]
        details = tag_details(tag, nodenames)
        copy_attribute(details, 'type', item)

        # Increment the position
        item['position'] = position
        # Ordinal should be the same as position in this case but set it anyway
        item['ordinal'] = tag_ordinal(tag)

        inline_graphics.append(item)

    return inline_graphics
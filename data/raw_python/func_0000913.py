def graphics(soup):
    """
    All graphic tags and some associated data about the related component doi
    and the parent of that doi (not always present), and whether it is
    part of a figure supplement
    """
    graphics = []

    graphic_tags = raw_parser.graphic(soup)

    position = 1

    for tag in graphic_tags:
        graphic_item = {}

        copy_attribute(tag.attrs, 'xlink:href', graphic_item, 'xlink_href')

        # Get the tag type
        nodenames = ["sub-article", "fig-group", "fig", "app"]
        details = tag_details(tag, nodenames)
        copy_attribute(details, 'type', graphic_item)

        parent_tag = first_parent(tag, nodenames)
        if parent_tag:
            details = tag_details(parent_tag, nodenames)
            copy_attribute(details, 'type', graphic_item, 'parent_type')
            copy_attribute(details, 'ordinal', graphic_item, 'parent_ordinal')
            copy_attribute(details, 'asset', graphic_item, 'parent_asset')
            copy_attribute(details, 'sibling_ordinal', graphic_item, 'parent_sibling_ordinal')
            copy_attribute(details, 'component_doi', graphic_item, 'parent_component_doi')

            # Try to get the parent parent - special for looking at fig tags
            #  use component_acting_parent_tag
            p_parent_tag = first_parent(parent_tag, nodenames)
            if p_parent_tag:
                acting_p_parent_tag = component_acting_parent_tag(p_parent_tag, parent_tag)
                if acting_p_parent_tag:
                    details = tag_details(acting_p_parent_tag, nodenames)
                    copy_attribute(details, 'type', graphic_item, 'p_parent_type')
                    copy_attribute(details, 'ordinal', graphic_item, 'p_parent_ordinal')
                    copy_attribute(details, 'asset', graphic_item, 'p_parent_asset')
                    copy_attribute(details, 'sibling_ordinal', graphic_item, 'p_parent_sibling_ordinal')
                    copy_attribute(details, 'component_doi', graphic_item, 'p_parent_component_doi')

        # Increment the position
        graphic_item['position'] = position
        # Ordinal should be the same as position in this case but set it anyway
        graphic_item['ordinal'] = tag_ordinal(tag)

        graphics.append(graphic_item)

        position += 1

    return graphics
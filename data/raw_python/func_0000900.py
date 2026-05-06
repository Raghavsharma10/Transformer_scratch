def copyright_holder_json(soup):
    "for json output add a full stop if ends in et al"
    holder = None
    permissions_tag = raw_parser.article_permissions(soup)
    if permissions_tag:
        holder = node_text(raw_parser.copyright_holder(permissions_tag))
    if holder is not None and holder.endswith('et al'):
        holder = holder + '.'
    return holder
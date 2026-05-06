def title_prefix(soup):
    "titlePrefix for article JSON is only articles with certain display_channel values"
    prefix = None
    display_channel_match_list = ['feature article', 'insight', 'editorial']
    for d_channel in display_channel(soup):
        if d_channel.lower() in display_channel_match_list:
            if raw_parser.sub_display_channel(soup):
                prefix = node_text(first(raw_parser.sub_display_channel(soup)))
    return prefix
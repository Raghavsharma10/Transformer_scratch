def display_channel(soup):
    """
    Find the subject areas of type display-channel
    """
    display_channel = []

    tags = raw_parser.display_channel(soup)
    for tag in tags:
        display_channel.append(node_text(tag))

    return display_channel
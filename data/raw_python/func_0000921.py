def all_contributors(soup, detail="brief"):
    "find all contributors not contrained to only the ones in article meta"
    contrib_tags = raw_parser.contributors(soup)
    contributors = format_authors(soup, contrib_tags, detail)
    return contributors
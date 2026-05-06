def full_research_organism(soup):
    "research-organism list including inline tags, such as italic"
    if not raw_parser.research_organism_keywords(soup):
        return []
    return list(map(node_contents_str, raw_parser.research_organism_keywords(soup)))
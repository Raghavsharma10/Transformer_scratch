def research_organism(soup):
    "Find the research-organism from the set of kwd-group tags"
    if not raw_parser.research_organism_keywords(soup):
        return []
    return list(map(node_text, raw_parser.research_organism_keywords(soup)))
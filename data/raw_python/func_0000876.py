def rewrite_json(rewrite_type, soup, json_content):
    """
    Due to XML content that will not conform with the strict JSON schema validation rules,
    for elife articles only, rewrite the JSON to make it valid
    """
    if not soup:
        return json_content
    if not elifetools.rawJATS.doi(soup) or not elifetools.rawJATS.journal_id(soup):
        return json_content

    # Hook only onto elife articles for rewriting currently
    journal_id_tag = elifetools.rawJATS.journal_id(soup)
    doi_tag = elifetools.rawJATS.doi(soup)
    journal_id = elifetools.utils.node_text(journal_id_tag)
    doi = elifetools.utils.doi_uri_to_doi(elifetools.utils.node_text(doi_tag))
    if journal_id.lower() == "elife":
        function_name = rewrite_function_name(journal_id, rewrite_type)
        if function_name:
            try:
                json_content = globals()[function_name](json_content, doi)
            except KeyError:
                pass
    return json_content
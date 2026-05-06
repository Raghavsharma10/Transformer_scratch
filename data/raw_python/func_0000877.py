def rewrite_elife_references_json(json_content, doi):
    """ this does the work of rewriting elife references json """
    references_rewrite_json = elife_references_rewrite_json()
    if doi in references_rewrite_json:
        json_content = rewrite_references_json(json_content, references_rewrite_json[doi])

    # Edge case delete one reference
    if doi == "10.7554/eLife.12125":
        for i, ref in enumerate(json_content):
            if ref.get("id") and ref.get("id") == "bib11":
                del json_content[i]

    return json_content
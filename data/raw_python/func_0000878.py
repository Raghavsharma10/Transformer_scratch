def rewrite_references_json(json_content, rewrite_json):
    """ general purpose references json rewriting by matching the id value """
    for ref in json_content:
        if ref.get("id") and ref.get("id") in rewrite_json:
            for key, value in iteritems(rewrite_json.get(ref.get("id"))):
                ref[key] = value
    return json_content
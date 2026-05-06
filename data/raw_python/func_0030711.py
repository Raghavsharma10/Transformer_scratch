def get(orcid_id):
    """
    Get an author based on an ORCID identifier.
    """
    resp = requests.get(ORCID_PUBLIC_BASE_URL + unicode(orcid_id),
                        headers=BASE_HEADERS)
    json_body = resp.json()
    return Author(json_body)
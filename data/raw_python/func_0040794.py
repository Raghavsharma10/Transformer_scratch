def journals_status(collection, raw=False):
    """
    This method retrieve the total of documents, articles (citable documents),
    issues and bibliografic references of a journal

    arguments
    collection: SciELO 3 letters Acronym
    issn: Journal ISSN

    return for journal context
    {
        "citable": 12140,
        "non_citable": 20,
        "docs": 12160,
        "issues": 120,
        "references": 286619
    }
    """

    tc = ThriftClient()

    body = {"query": {"filtered": {}}}

    fltr = {}

    query = {
        "query": {
            "bool": {
                "must": [
                    {
                        "match": {
                            "collection": collection
                        }
                    }
                ]
            }
        }
    }

    body['query']['filtered'].update(fltr)
    body['query']['filtered'].update(query)

    query_parameters = [
        ('size', '0'),
        ('search_type', 'count')
    ]

    body['aggs'] = {
        "status": {
            "terms": {
                "field": "status"
            }
        }
    }

    query_result = tc.search('journal', json.dumps(body), query_parameters)

    computed = _compute_journals_status(query_result)

    return query_result if raw else computed
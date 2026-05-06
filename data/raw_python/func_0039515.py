def number_of_issues_by_year(self, issn, collection, years=0, type=None):
        """
        type: ['regular', 'supplement', 'pressrelease', 'ahead', 'special']
        """

        body = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "match": {
                                "issn": issn
                            }
                        },
                        {
                            "match": {
                                "collection": collection
                            }
                        }
                    ]
                }
            },
            "aggs": {
                "issue": {
                    "cardinality": {
                        "field": "issue"
                    }
                }
            }

        }

        if type:
            body['query']['bool']['must'].append({"match": {"issue_type": type}})

        if years != 0:
            body['aggs'] = {
                "publication_year": {
                    "terms": {
                        "field": "publication_year",
                        "size": years,
                        "order": {
                            "_term": 'desc'
                        }
                    },
                    "aggs": {
                        "issue": {
                            "cardinality": {
                                "field": "issue"
                            }
                        }
                    }
                }
            }

        query_parameters = [
            ('size', '0')
        ]

        query_result = self.search(
            'article', json.dumps(body), query_parameters
        )

        return self._compute_number_of_issues_by_year(
            query_result, years=years)
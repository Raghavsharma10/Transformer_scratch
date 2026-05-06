def _fuzzy_custom_query(issn, titles):
        """
            Este metodo constroi a lista de filtros por título de periódico que
            será aplicada na pesquisa boleana como match por similaridade "should".
            A lista de filtros é coletada do template de pesquisa customizada
            do periódico, quanto este template existir.
        """
        custom_queries = journal_titles.load(issn).get('should', [])
        titles = [{'title': i} for i in titles if i not in [x['title'] for x in custom_queries]]
        titles.extend(custom_queries)

        for item in titles:

            if len(item['title'].strip()) == 0:
                continue

            query = {
                "fuzzy": {
                    "reference_source_cleaned": {
                        "value": utils.cleanup_string(item['title']),
                        "fuzziness": item.get('fuzziness', 3),
                        "max_expansions": 50
                    }
                }
            }

            yield query
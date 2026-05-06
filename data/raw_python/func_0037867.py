def _parse_response(self, results):
        """
        Parses result dictionary into a SolrResults object
        """

        dict_response = results.get("response")
        result_obj = SolrResults()
        result_obj.query_time = results.get("responseHeader").get("QTime", None)
        result_obj.results_count = dict_response.get("numFound", 0)
        result_obj.start_index = dict_response.get("start", 0)

        for doc in dict_response.get("docs", []):
            result_obj.documents.append(doc)

        # Process facets
        if "facet_counts" in results:
            facet_types = ["facet_fields", "facet_dates", "facet_ranges", "facet_queries"]
            for type in facet_types:
                assert type in results.get("facet_counts")
                items = results.get("facet_counts").get(type)
                for field, values in items.items():
                    result_obj.facets[field] = []

                    # Range facets have results in "counts" subkey and "between/after" on top level. Flatten this.
                    if type == "facet_ranges":
                        if not "counts" in values:
                            continue

                        for facet, value in values["counts"].items():
                            result_obj.facets[field].append((facet, value))

                        if "before" in values:
                            result_obj.facets[field].append(("before", values["before"]))

                        if "after" in values:
                            result_obj.facets[field].append(("after", values["after"]))
                    else:
                        for facet, value in values.items():
                            # Date facets have metadata fields between the results, skip the params, keep "before" and "after" fields for other
                            if type == "facet_dates" and \
                            (facet == "gap" or facet == "between" or facet == "start" or facet == "end"):
                                continue
                            result_obj.facets[field].append((facet, value))

        # Process highlights
        if "highlighting" in results:
            for key, value in results.get("highlighting").items():
                result_obj.highlights[key] = value

        return result_obj
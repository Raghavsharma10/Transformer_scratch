def get_datasets(self, query='*:*', **kwargs):
        # type: (str, Any) -> List[hdx.data.dataset.Dataset]
        """Get list of datasets in organization

        Args:
            query (str): Restrict datasets returned to this query (in Solr format). Defaults to '*:*'.
            **kwargs: See below
            sort (string): Sorting of the search results. Defaults to 'relevance asc, metadata_modified desc'.
            rows (int): Number of matching rows to return. Defaults to all datasets (sys.maxsize).
            start (int): Offset in the complete result for where the set of returned datasets should begin
            facet (string): Whether to enable faceted results. Default to True.
            facet.mincount (int): Minimum counts for facet fields should be included in the results
            facet.limit (int): Maximum number of values the facet fields return (- = unlimited). Defaults to 50.
            facet.field (List[str]): Fields to facet upon. Default is empty.
            use_default_schema (bool): Use default package schema instead of custom schema. Defaults to False.

        Returns:
            List[Dataset]: List of datasets in organization
        """
        return hdx.data.dataset.Dataset.search_in_hdx(query=query,
                                                      configuration=self.configuration,
                                                      fq='organization:%s' % self.data['name'], **kwargs)
def clear_solr(self, catalog="hypermap"):
        """Clear all indexes in the solr core"""
        solr_url = "{0}/solr/{1}".format(SEARCH_URL, catalog)
        solr = pysolr.Solr(solr_url, timeout=60)
        solr.delete(q='*:*')
        LOGGER.debug('Solr core cleared')
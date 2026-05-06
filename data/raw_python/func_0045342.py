def riak_http_search_query(self, solr_core, solr_params, count_deleted=False):
        """
        This method is for advanced SOLR queries. Riak HTTP search query endpoint,
        sends solr_params and query string as a proxy and returns solr reponse.
        
        Args:
            solr_core (str): solr core on which query will be executed
            
            solr_params (str): solr specific query params, such as rows, start, fl, df, wt etc..
            
            count_deleted (bool): ignore deleted records or not 
        
        Returns:
            (dict): dict of solr response
        
        """

        # append current _solr_query params
        sq = ["%s%%3A%s" % (q[0], q[1]) for q in self._solr_query]
        if not count_deleted:
            sq.append("-deleted%3ATrue")

        search_host = "http://%s:%s/search/query/%s?wt=json&q=%s&%s" % (
            settings.RIAK_SERVER,
            settings.RIAK_HTTP_PORT,
            solr_core,
            "+AND+".join(sq),
            solr_params
        )

        return json.loads(bytes_to_str(urlopen(search_host).read()))
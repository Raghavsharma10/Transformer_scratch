def api(self, action, args=None):
        """
        Sends a request to Solr Collections API.
        Documentation is here: https://cwiki.apache.org/confluence/display/solr/Collections+API

        :param string action: Name of the collection for the action
        :param dict args: Dictionary of specific parameters for action
        """
        if args is None:
            args = {}
        args['action'] = action.upper()

        try:
            res, con_info = self.solr.transport.send_request(endpoint='admin/collections', params=args)
        except Exception as e:
            self.logger.error("Error querying SolrCloud Collections API. ")
            self.logger.exception(e)
            raise e

        if 'responseHeader' in res and res['responseHeader']['status'] == 0:
            return res, con_info
        else:
            raise SolrError("Error Issuing Collections API Call for: {} +".format(con_info, res))
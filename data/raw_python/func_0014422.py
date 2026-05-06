def _retry(function):
        """
        Internal mechanism to try to send data to multiple Solr Hosts if
        the query fails on the first one.
        """

        def inner(self, **kwargs):
            last_exception = None
            #for host in self.router.get_hosts(**kwargs):
            for host in self.host:
                try:
                    return function(self, host, **kwargs)
                except SolrError as e:
                    self.logger.exception(e)
                    raise
                except ConnectionError as e:
                    self.logger.exception("Tried connecting to Solr, but couldn't because of the following exception.")
                    if '401' in e.__str__():
                        raise
                    last_exception = e
            # raise the last exception after contacting all hosts instead of returning None
            if last_exception is not None:
                raise last_exception
        return inner
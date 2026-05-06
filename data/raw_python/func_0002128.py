def get_url(self, datatype, verb, urltype, params={}, api_host=None, api_version=None):
        """Returns a fully formed url

        :param datatype: a string identifying the data the url will access.
        :param verb: the HTTP verb needed for use with the url.
        :param urltype: an adjective used to the nature of the request.
        :param \*\*params: substitution variables for the URL.
        :return: string
        :rtype: A fully formed url.
        """
        api_version = api_version or 'v1'
        api_host = api_host or self.host

        subst = params.copy()
        subst['api_host'] = api_host
        subst['api_version'] = api_version

        url = "https://{api_host}/services/api/{api_version}"
        url += self.get_url_path(datatype, verb, urltype, params, api_version)
        return url.format(**subst)
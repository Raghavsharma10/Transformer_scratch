def reverse_url(self, datatype, url, verb='GET', urltype='single', api_version=None):
        """
        Extracts parameters from a populated URL

        :param datatype: a string identifying the data the url accesses.
        :param url: the fully-qualified URL to extract parameters from.
        :param verb: the HTTP verb needed for use with the url.
        :param urltype: an adjective used to the nature of the request.
        :return: dict
        """
        api_version = api_version or 'v1'
        templates = getattr(self, 'URL_TEMPLATES__%s' % api_version)

        # this is fairly simplistic, if necessary we could use the parse lib
        template_url = r"https://(?P<api_host>.+)/services/api/(?P<api_version>.+)"
        template_url += re.sub(r'{([^}]+)}', r'(?P<\1>.+)', templates[datatype][verb][urltype])
        # /foo/{foo_id}/bar/{id}/
        m = re.match(template_url, url or '')
        if not m:
            raise KeyError("No reverse match from '%s' to %s.%s.%s" % (url, datatype, verb, urltype))

        r = m.groupdict()
        del r['api_host']
        if r.pop('api_version') != api_version:
            raise ValueError("API version mismatch")
        return r
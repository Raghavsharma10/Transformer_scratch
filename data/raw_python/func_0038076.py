def _build_cache_key(request):
        """
        Generated the key name used to cache responses
        :param request: request used to retrieve API response
        :return: formatted cache name
        """
        str_hash = md5(
            (request.method + request.url + str(request.params) + str(request.data) + str(request.json)).encode(
                'utf-8')).hexdigest()
        return 'esi_%s' % str_hash
def do_request(self, json_obj):
        """Perform one HTTP request to Zabbix API"""
        self.debug('Request: url="%s" headers=%s', self._api_url, self._http_headers)
        self.debug('Request: body=%s', json_obj)
        self.r_query.append(json_obj)

        request = urllib2.Request(url=self._api_url, data=json_obj.encode('utf-8'), headers=self._http_headers)
        opener = urllib2.build_opener(self._http_handler)
        urllib2.install_opener(opener)

        try:
            response = opener.open(request, timeout=self.timeout)
        except Exception as e:
            raise ZabbixAPIException('HTTP connection problem: %s' % e)

        self.debug('Response: code=%s', response.code)

        # NOTE: Getting a 412 response code means the headers are not in the list of allowed headers.
        if response.code != 200:
            raise ZabbixAPIException('HTTP error %s: %s' % (response.status, response.reason))

        reads = response.read()

        if len(reads) == 0:
            raise ZabbixAPIException('Received zero answer')

        try:
            jobj = json.loads(reads.decode('utf-8'))
        except ValueError as e:
            self.log(ERROR, 'Unable to decode. returned string: %s', reads)
            raise ZabbixAPIException('Unable to decode response: %s' % e)

        self.debug('Response: body=%s', jobj)
        self.id += 1

        if 'error' in jobj:  # zabbix API error
            error = jobj['error']

            if isinstance(error, dict):
                raise ZabbixAPIError(**error)

        try:
            return jobj['result']
        except KeyError:
            raise ZabbixAPIException('Missing result in API response')
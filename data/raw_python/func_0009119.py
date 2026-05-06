def convert_plugin_def(http_method, funcs):
        """
        This function parses one of the elements of the definitions dict for a
        plugin and extracts the relevant information

        :param http_method: HTTP method that uses (GET, POST, DELETE, ...)
        :param funcs: functions related to that HTTP method
        """
        methods = []
        if http_method not in ('GET', 'PUT', 'POST', 'DELETE'):
            logger.error(
                'Plugin load failure, HTTP method %s unsupported.',
                http_method,
            )
            return methods
        for fname, params in six.iteritems(funcs):
            method = {
                'apis': [{'short_description': 'no-doc'}],
                'params': [],
            }
            method['apis'][0]['http_method'] = http_method
            method['apis'][0]['api_url'] = '/api/' + fname
            method['name'] = fname
            for pname, pdef in six.iteritems(params):
                param = {
                    'name': pname,
                    'validator': "Must be %s" % pdef['ptype'],
                    'description': '',
                    'required': pdef['required'],
                }
                method['params'].append(param)
            methods.append(method)
        return methods
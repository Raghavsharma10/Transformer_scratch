def call_on_each_endpoint(self, callback):
        """Find all server endpoints defined in the swagger spec and calls 'callback' for each,
        with an instance of EndpointData as argument.
        """

        if 'paths' not in self.swagger_dict:
            return

        for path, d in list(self.swagger_dict['paths'].items()):
            for method, op_spec in list(d.items()):
                data = EndpointData(path, method)

                # Which server method handles this endpoint?
                if 'x-bind-server' not in op_spec:
                    if 'x-no-bind-server' in op_spec:
                        # That route should not be auto-generated
                        log.info("Skipping generation of %s %s" % (method, path))
                        continue
                    else:
                        raise Exception("Swagger api defines no x-bind-server for %s %s" % (method, path))
                data.handler_server = op_spec['x-bind-server']

                # Make sure that endpoint only produces 'application/json'
                if 'produces' not in op_spec:
                    raise Exception("Swagger api has no 'produces' section for %s %s" % (method, path))
                if len(op_spec['produces']) != 1:
                    raise Exception("Expecting only one type under 'produces' for %s %s" % (method, path))
                if op_spec['produces'][0] == 'application/json':
                    data.produces_json = True
                elif op_spec['produces'][0] == 'text/html':
                    data.produces_html = True
                else:
                    raise Exception("Only 'application/json' or 'text/html' are supported. See %s %s" % (method, path))

                # Which client method handles this endpoint?
                if 'x-bind-client' in op_spec:
                    data.handler_client = op_spec['x-bind-client']

                # Should we decorate the server handler?
                if 'x-decorate-server' in op_spec:
                    data.decorate_server = op_spec['x-decorate-server']

                # Should we manipulate the requests parameters?
                if 'x-decorate-request' in op_spec:
                    data.decorate_request = op_spec['x-decorate-request']

                # Generate a bravado-core operation object
                data.operation = Operation.from_spec(self.spec, path, method, op_spec)

                # Figure out how parameters are passed: one json in body? one or
                # more values in query?
                if 'parameters' in op_spec:
                    params = op_spec['parameters']
                    for p in params:
                        if p['in'] == 'body':
                            data.param_in_body = True
                        if p['in'] == 'query':
                            data.param_in_query = True
                        if p['in'] == 'path':
                            data.param_in_path = True

                    if data.param_in_path:
                        # Substitute {...} with <...> in path, to make a Flask friendly path
                        data.path = data.path.replace('{', '<').replace('}', '>')

                    if data.param_in_body and data.param_in_query:
                        raise Exception("Cannot support params in both body and param (%s %s)" % (method, path))

                else:
                    data.no_params = True

                callback(data)
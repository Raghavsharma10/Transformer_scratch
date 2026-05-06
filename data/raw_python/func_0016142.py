def json_obj(self, method, params=None, auth=True):
        """Return JSON object expected by the Zabbix API"""
        if params is None:
            params = {}

        obj = {
            'jsonrpc': '2.0',
            'method': method,
            'params': params,
            'auth': self.__auth if auth else None,
            'id': self.id,
        }

        return json.dumps(obj)
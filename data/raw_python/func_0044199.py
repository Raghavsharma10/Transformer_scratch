def post(self):
        """Accepts jsorpc post request.
        Retrieves data from request body.
        """
        # type(data) = dict
        data = json.loads(self.request.body.decode())
        # type(method) = str
        method = data["method"]
        # type(params) = dict
        params = data["params"]
        if method == "sendmail":
            response = dispatch([sendmail],{'jsonrpc': '2.0', 'method': 'sendmail', 'params': [params], 'id': 1})
            #self.write(response)
        else:
            pass
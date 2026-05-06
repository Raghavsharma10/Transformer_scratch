def outputjson(self, obj):
        """
        Serialize `obj` with JSON and output to the client
        """
        self.header('Content-Type', 'application/json')
        self.outputdata(json.dumps(obj).encode('ascii'))
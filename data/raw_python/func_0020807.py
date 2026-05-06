def __callServer(self, method="", params={}, data={}, callmethod='GET', content='application/json'):
        """
        A private method to make HTTP call to the DBS Server

        :param method: REST API to call, e.g. 'datasets, blocks, files, ...'.
        :type method: str
        :param params: Parameters to the API call, e.g. {'dataset':'/PrimaryDS/ProcessedDS/TIER'}.
        :type params: dict
        :param callmethod: The HTTP method used, by default it is HTTP-GET, possible values are GET, POST and PUT.
        :type callmethod: str
        :param content: The type of content the server is expected to return. DBS3 only supports application/json
        :type content: str

        """
        UserID = os.environ['USER']+'@'+socket.gethostname()
        try:
            UserAgent = "DBSClient/"+os.environ['DBS3_CLIENT_VERSION']+"/"+ self.userAgent
        except:
            UserAgent = "DBSClient/Unknown"+"/"+ self.userAgent
        request_headers =  {"Content-Type": content, "Accept": content, "UserID": UserID, "User-Agent":UserAgent }

        method_func = getattr(self.rest_api, callmethod.lower())

        data = cjson.encode(data)

        try:
            self.http_response = method_func(self.url, method, params, data, request_headers)
        except HTTPError as http_error:
            self.__parseForException(http_error)

        if content != "application/json":
            return self.http_response.body

        try:
            json_ret=cjson.decode(self.http_response.body)
        except cjson.DecodeError:
            print("The server output is not a valid json, most probably you have a typo in the url.\n%s.\n" % self.url, file=sys.stderr)
            raise dbsClientException("Invalid url", "Possible urls are %s" %self.http_response.body)

        return json_ret
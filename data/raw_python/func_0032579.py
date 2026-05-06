def handle_single_request(self, request_object):
        """
        Handles a single request object and returns the raw response

        :param request_object:
        """
        if not isinstance(request_object, (MethodCall, Notification)):
            raise TypeError("Invalid type for request_object")

        method_name = request_object.method_name
        params = request_object.params
        req_id = request_object.id

        request_body = self.build_request_body(method_name, params, id=req_id)
        http_request = self.build_http_request_obj(request_body)

        try:
            response = urllib.request.urlopen(http_request)
        except urllib.request.HTTPError as e:
            raise CalledServiceError(e)

        if not req_id:
            return

        response_body = json.loads(response.read().decode())
        return response_body
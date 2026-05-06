def write_single_response(self, response_obj):
        """
        Writes a json rpc response ``{"result": result, "error": error, "id": id}``.
        If the ``id`` is ``None``, the response will not contain an ``id`` field.
        The response is sent to the client as an ``application/json`` response. Only one call per
        response is allowed

        :param response_obj: A Json rpc response object
        :return:
        """
        if not isinstance(response_obj, JsonRpcResponse):
            raise ValueError(
                "Expected JsonRpcResponse, but got {} instead".format(type(response_obj).__name__))

        if not self.response_is_sent:
            self.set_status(200)
            self.set_header("Content-Type", "application/json")
            self.finish(response_obj.to_string())
            self.response_is_sent = True
def notify(self, method_name_or_object, params=None):
        """
        Sends a notification to the service by calling the ``method_name``
        method with the ``params`` parameters. Does not wait for a response, even
        if the response triggers an error.

        :param method_name_or_object: the name of the method to be called or a ``Notification``
                                      instance
        :param params: a list of dict representing the parameters for the call
        :return: None
        """
        if isinstance(method_name_or_object, Notification):
            req_obj = method_name_or_object
        else:
            req_obj = Notification(method_name_or_object, params)
        self.handle_single_request(req_obj)
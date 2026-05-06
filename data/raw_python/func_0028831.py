def default_route_options():
        """
        Default callback for OPTIONS request
        :rtype: Response
        """
        response_obj = OrderedDict()

        response_obj["status"] = True
        response_obj["data"] = "Ok"

        return Response(response_obj, content_type="application/json", charset="utf-8")
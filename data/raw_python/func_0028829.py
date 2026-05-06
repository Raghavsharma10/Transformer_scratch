def no_route_found(self, request):
        """
        Default callback for route not found
        :param request HttpRequest
        :rtype: Response
        """
        response_obj = OrderedDict()
        response_obj["status"] = False
        response_obj["exceptions"] = {
            "message": "No route found for {0} {1}".format(self.__method, self.__uri),
        }
        response_obj["request"] = {
            "method": self.__method,
            "path_info": self.__uri,
            "content": request.body.decode("utf-8")
        }
        response_obj["message"] = "We are sorry, but something went terribly wrong."

        return Response(response_obj, content_type="application/json", status=404, charset="utf-8")
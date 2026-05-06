def handle(self, request: HttpRequest) -> HttpResponse:
        """
        Prepares for the CallBackResolver and handles the response and exceptions
        :param request HttpRequest
        :rtype: HttpResponse
        """
        self.__request_start = datetime.now()
        self.__request = request
        self.__uri = request.path[1:]
        self.__method = request.method

        # Initializes the callable controller and call it's connect method to get the mapped end-points.
        controller: RouteMapping = self.__controller().connect(self.app)

        self.__end_points = controller.get_routes()

        indent = self.get_json_ident(request.META)

        if self.set_end_point_uri() is False:
            return self.set_response_headers(self.no_route_found(self.__request).render(indent))

        response = HttpResponse(None)
        try:
            response = self.exec_route_callback()
        except RinzlerHttpException as e:
            client.captureException()
            self.app.log.error(f"< {e.status_code}", exc_info=True)
            response = Response(None, status=e.status_code)
        except RequestDataTooBig:
            client.captureException()
            self.app.log.error("< 413", exc_info=True)
            response = Response(None, status=413)
        except BaseException:
            client.captureException()
            self.app.log.error("< 500", exc_info=True)
            response = Response(None, status=500)
        finally:
            if type(response) == Response:
                return self.set_response_headers(response.render(indent))
            else:
                return self.set_response_headers(response)
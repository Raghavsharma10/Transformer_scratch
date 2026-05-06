def dispatch(self, request, environ):
        """
        Checks which Grant supports the current request and dispatches to it.

        :param request: The incoming request.
        :type request: :class:`oauth2.web.Request`
        :param environ: Dict containing variables of the environment.
        :type environ: dict

        :return: An instance of ``oauth2.web.Response``.
        """
        try:
            grant_type = self._determine_grant_type(request)

            response = self.response_class()

            grant_type.read_validate_params(request)

            return grant_type.process(request, response, environ)
        except OAuthInvalidNoRedirectError:
            response = self.response_class()
            response.add_header("Content-Type", "application/json")
            response.status_code = 400
            response.body = json.dumps({
                "error": "invalid_redirect_uri",
                "error_description": "Invalid redirect URI"
            })

            return response
        except OAuthInvalidError as err:
            response = self.response_class()
            return grant_type.handle_error(error=err, response=response)
        except UnsupportedGrantError:
            response = self.response_class()
            response.add_header("Content-Type", "application/json")
            response.status_code = 400
            response.body = json.dumps({
                "error": "unsupported_response_type",
                "error_description": "Grant not supported"
            })

            return response
        except:
            app_log.error("Uncaught Exception", exc_info=True)
            response = self.response_class()
            return grant_type.handle_error(
                error=OAuthInvalidError(error="server_error",
                                        explanation="Internal server error"),
                response=response)
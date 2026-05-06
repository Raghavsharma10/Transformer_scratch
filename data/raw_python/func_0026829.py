def respond_webhook(self, environ):
        """
        Passes the request onto a bot with a webhook if the webhook
        path is requested.
        """
        request = FieldStorage(fp=environ["wsgi.input"], environ=environ)
        url = environ["PATH_INFO"]
        params = dict([(k, request[k].value) for k in request])
        try:
            if self.bot is None:
                raise NotImplementedError
            response = self.bot.handle_webhook_event(environ, url, params)
        except NotImplementedError:
            return 404
        except:
            self.logger.debug(format_exc())
            return 500
        return response or 200
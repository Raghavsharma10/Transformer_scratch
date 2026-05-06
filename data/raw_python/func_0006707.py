def on_post(self):
        """Extracts the request, feeds the module, and returns the response."""
        request = self.environ['wsgi.input']
        try:
            return self.process_request(request)
        except ClientError as exc:
            return self.on_client_error(exc)
        except BadGateway as exc:
            return self.on_bad_gateway(exc)
        except InvalidConfig:
            raise
        except Exception as exc: # pragma: no cover # pylint: disable=W0703
            logging.error('Unknown exception: ', exc_info=exc)
            return self.on_internal_error()
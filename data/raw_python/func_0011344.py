def process_response(self, req, resp, resource):
        """Post-processing of the response (after routing).

        Args:
            req: Request object.
            resp: Response object.
            resource: Resource object to which the request was
                routed. May be None if no route was found
                for the request.
        """
        if isinstance(resp.body, dict):
            try:
                resp.body = json.dumps(resp.body)
            except(nameError):
                resp.status = falcon.HTTP_500
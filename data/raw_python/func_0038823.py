def render_to_response(self, context, **response_kwargs):
        """
        This endpoint sets very permiscuous CORS headers.

        Access-Control-Allow-Origin is set to the request Origin. This allows
          a page from ANY domain to make a request to this endpoint.

        Access-Control-Allow-Credentials is set to true. This allows requesting
          poll data in our authenticated test/staff environments.

        This particular combination of headers means this endpoint is a potential
            CSRF target.

        This enpoint MUST NOT write data. And it MUST NOT return any sensitive data.
        """
        serializer = PollPublicSerializer(self.object)
        response = HttpResponse(
            json.dumps(serializer.data),
            content_type="application/json"
        )
        if "HTTP_ORIGIN" in self.request.META:
            response["Access-Control-Allow-Origin"] = self.request.META["HTTP_ORIGIN"]
            response["Access-Control-Allow-Credentials"] = 'true'

        return response
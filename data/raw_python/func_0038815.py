async def make_response(self, request, response, **response_kwargs):
        """Convert a handler result to web response."""
        while iscoroutine(response):
            response = await response

        if isinstance(response, StreamResponse):
            return response

        response_kwargs.setdefault('content_type', 'application/json')

        return Response(text=dumps(response), **response_kwargs)
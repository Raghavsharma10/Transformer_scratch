def render(self, request=None, context=None, template_name=None):
        """
        Returns a HttpResponse of the right media type as specified by the
        request.
        
        context can contain status_code and additional_headers members, to set
        the HTTP status code and headers of the request, respectively.
        template_name should lack a file-type suffix (e.g. '.html', as
        renderers will append this as necessary.
        """
        request, context, template_name = self.get_render_params(request, context, template_name)

        self.set_renderers()

        status_code = context.pop('status_code', http_client.OK)
        additional_headers = context.pop('additional_headers', {})

        for renderer in request.renderers:
            response = renderer(request, context, template_name)
            if response is NotImplemented:
                continue
            response.status_code = status_code
            response.renderer = renderer
            break
        else:
            tried_mimetypes = list(itertools.chain(*[r.mimetypes for r in request.renderers]))
            response = self.http_not_acceptable(request, tried_mimetypes)
            response.renderer = None
        for key, value in additional_headers.items():
            response[key] = value

        # We're doing content-negotiation, so tell the user-agent that the
        # response will vary depending on the accept header.
        patch_vary_headers(response, ('Accept',))
        return response
def _match(self, request, response):
        """Match all requests/responses that satisfy the following conditions:

        * An Admin App; i.e. the path is something like /admin/some_app/
        * The ``include_flag`` is not in the response's content

        """
        is_html = 'text/html' in response.get('Content-Type', '')
        if is_html and hasattr(response, 'rendered_content'):
            correct_path = PATH_MATCHER.match(request.path) is not None
            not_included = self.include_flag not in response.rendered_content
            return correct_path and not_included
        return False
def get_static(self, _, file_name=None):
        """
        Get static content for UI.
        """
        content_type = {
            'ss': 'text/css',
            'js': 'application/javascript',
        }.get(file_name[-2:])
        if not content_type:
            raise HttpError(HTTPStatus.NOT_FOUND, 42)

        return HttpResponse(self.load_static(file_name), headers={
            'Content-Type': content_type,
            'Content-Encoding': 'gzip',
            'Cache-Control': 'public, max-age=300',
        })
def document(self):
        """Render the error document"""
        resp = request.environ.get('pylons.original_response')
        page = error_document_template % \
            dict(prefix=request.environ.get('SCRIPT_NAME', ''),
                 code=request.params.get('code', resp.status_int),
                 message=request.params.get('message', resp.body))
        return page
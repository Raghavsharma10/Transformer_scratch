def paste_traceback(self, request, traceback):
        """Paste the traceback and return a JSON response."""
        rv = traceback.paste()
        return Response(json.dumps(rv), content_type='application/json')
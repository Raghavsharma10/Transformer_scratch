def get_ui(self, _):
        """
        Load the Swagger UI interface
        """
        if not self._ui_cache:
            content = self.load_static('ui.html')
            if isinstance(content, binary_type):
                content = content.decode('UTF-8')
            self._ui_cache = content.replace(u"{{SWAGGER_PATH}}", str(self.swagger_path))
        return HttpResponse(self._ui_cache, headers={
            'Content-Type': 'text/html'
        })
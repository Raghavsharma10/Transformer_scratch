def get_swagger(self, request):
        """
        Generate this document.
        """
        api_base = self.parent
        paths, definitions = self.parse_operations()
        codecs = getattr(self.cenancestor, 'registered_codecs', CODECS)  # type: dict
        return dict_filter({
            'swagger': '2.0',
            'info': {
                'title': self.title,
                'version': str(getattr(api_base, 'version', 0))
            },
            'host': self.host or request.host,
            'schemes': list(self.schemes) or None,
            'basePath': str(self.base_path),
            'consumes': list(codecs.keys()),
            'produces': list(codecs.keys()),
            'paths': paths,
            'definitions': definitions,
            'securityDefinitions': self.security_definitions(),
        })
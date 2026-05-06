def build_object(self, obj):
        """Override django-bakery to skip profiles that raise 404"""
        try:
            build_path = self.get_build_path(obj)
            self.request = self.create_request(build_path)
            self.request.user = AnonymousUser()
            self.set_kwargs(obj)
            self.build_file(build_path, self.get_content())
        except Http404:
            # cleanup directory
            self.unbuild_object(obj)
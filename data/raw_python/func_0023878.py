def build_queryset(self):
        """Override django-bakery's build logic to fake pagination."""
        paths = [(os.path.join(self.build_prefix, 'index.html'), {})]
        self.request = None
        queryset = self.get_queryset()
        paginator = self.get_paginator(queryset, self.get_paginate_by(queryset))
        for page in paginator.page_range:
            paths.append(
                (os.path.join(self.build_prefix, 'page',
                              '%d' % page, 'index.html'), {'page': page}))
        for build_path, kwargs in paths:
            self.request = self.create_request(build_path)
            # Add a user with no permissions
            self.request.user = AnonymousUser()
            # Fake context so views work as expected
            self.kwargs = kwargs
            self.prep_directory(build_path)
            target_path = os.path.join(settings.BUILD_DIR, build_path)
            self.build_file(target_path, self.get_content())
def as_view(self, view, cacheable=False, extra_permission=None):
        """
        Wraps a view in authentication/caching logic

        extra_permission can be used to require an extra permission for this view, such as a module permission
        """
        def inner(request, *args, **kwargs):
            if not self.has_permission(request, extra_permission):
                # show login pane
                return self.login(request)
            return view(request, *args, **kwargs)

        # Mark it as never_cache
        if not cacheable:
            inner = never_cache(inner)

        # We add csrf_protect here so this function can be used as a utility
        # function for any view, without having to repeat 'csrf_protect'.
        if not getattr(view, 'csrf_exempt', False):
            inner = csrf_protect(inner)

        inner = ensure_csrf_cookie(inner)

        return update_wrapper(inner, view)
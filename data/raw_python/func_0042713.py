def logout(self, request, extra_context=None):
        """
        Logs out the user for the given HttpRequest.

        This should *not* assume the user is already logged in.
        """
        from django.contrib.auth.views import logout
        defaults = {
            'extra_context': extra_context or {},
            'template_name': 'cms/logged_out.html',
        }
        if self.logout_template is not None:
            defaults['template_name'] = self.logout_template
        return logout(request, **defaults)
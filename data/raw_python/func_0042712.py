def password_change_done(self, request, extra_context=None):
        """
        Displays the "success" page after a password change.
        """
        from django.contrib.auth.views import password_change_done
        defaults = {
            'extra_context': extra_context or {},
            'template_name': 'cms/password_change_done.html',
        }
        if self.password_change_done_template is not None:
            defaults['template_name'] = self.password_change_done_template
        return password_change_done(request, **defaults)
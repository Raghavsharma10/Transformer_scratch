def password_change(self, request):
        """
        Handles the "change password" task -- both form display and validation.

        Uses the default auth views.
        """
        from django.contrib.auth.views import password_change
        url = reverse('admin:cms_password_change_done')
        defaults = {
            'post_change_redirect': url,
            'template_name': 'cms/password_change_form.html',
        }
        if self.password_change_template is not None:
            defaults['template_name'] = self.password_change_template
        return password_change(request, **defaults)
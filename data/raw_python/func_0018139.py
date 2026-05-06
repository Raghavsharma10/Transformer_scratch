def get_permissions_for_registration(self):
        """
        Utilised by Wagtail's 'register_permissions' hook to allow permissions
        for a model to be assigned to groups in settings. This is only required
        if the model isn't a Page model, and isn't registered as a Snippet
        """
        from wagtail.wagtailsnippets.models import SNIPPET_MODELS
        if not self.is_pagemodel and self.model not in SNIPPET_MODELS:
            return self.permission_helper.get_all_model_permissions()
        return Permission.objects.none()
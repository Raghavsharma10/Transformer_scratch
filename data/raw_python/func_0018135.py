def confirm_delete_view(self, request, object_id):
        """
        Instantiates a class-based view to provide 'delete confirmation'
        functionality for the assigned model, or redirect to Wagtail's delete
        confirmation view if the assigned model extends 'Page'. The view class
        used can be overridden by changing the 'confirm_delete_view_class'
        attribute.
        """
        kwargs = {'model_admin': self, 'object_id': object_id}
        view_class = self.confirm_delete_view_class
        return view_class.as_view(**kwargs)(request)
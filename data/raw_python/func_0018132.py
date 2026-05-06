def create_view(self, request):
        """
        Instantiates a class-based view to provide 'creation' functionality for
        the assigned model, or redirect to Wagtail's create view if the
        assigned model extends 'Page'. The view class used can be overridden by
        changing the 'create_view_class' attribute.
        """
        kwargs = {'model_admin': self}
        view_class = self.create_view_class
        return view_class.as_view(**kwargs)(request)
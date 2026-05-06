def choose_parent_view(self, request):
        """
        Instantiates a class-based view to provide a view that allows a parent
        page to be chosen for a new object, where the assigned model extends
        Wagtail's Page model, and there is more than one potential parent for
        new instances. The view class used can be overridden by changing the
        'choose_parent_view_class' attribute.
        """
        kwargs = {'model_admin': self}
        view_class = self.choose_parent_view_class
        return view_class.as_view(**kwargs)(request)
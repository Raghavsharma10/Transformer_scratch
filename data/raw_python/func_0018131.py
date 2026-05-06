def index_view(self, request):
        """
        Instantiates a class-based view to provide listing functionality for
        the assigned model. The view class used can be overridden by changing
        the 'index_view_class' attribute.
        """
        kwargs = {'model_admin': self}
        view_class = self.index_view_class
        return view_class.as_view(**kwargs)(request)
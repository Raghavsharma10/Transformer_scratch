def get_initialized_view_and_name(self, view_name,
                                    follow_parent=True, **extra_kwargs):
        """
        Creates and returns a new instance of a CMSView \
        and it's url_name.

        :param view_name: The name of the view to return.
        :param follow_parent: If we encounter a parent reference should \
        we follow it. Defaults to True.
        :param extra_kwargs: Keyword arguments to pass to the view.
        """

        view, name = self.get_view_and_name(view_name)

        # Initialize the view with the right kwargs
        if hasattr(view, 'as_view'):
            e = dict(extra_kwargs)
            e.update(**self._get_view_kwargs(view, view_name))
            e['name'] = view_name
            view = view(**e)

        # It is a Bundle return the main
        elif isinstance(view, Bundle):
            view, name = view.get_initialized_view_and_name('main',
                                                        **extra_kwargs)
        elif view == self.parent_attr and self.parent:
            if follow_parent:
                return self.parent.get_initialized_view_and_name(view_name,
                                                              **extra_kwargs)
            else:
                view = None
                name = None
        return view, name
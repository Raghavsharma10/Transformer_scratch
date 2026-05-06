def get_view_url(self, view_name, user,
                     url_kwargs=None, context_kwargs=None,
                     follow_parent=True, check_permissions=True):
        """
        Returns the url for a given view_name. If the view isn't
        found or the user does not have permission None is returned.
        A NoReverseMatch error may be raised if the view was unable
        to find the correct keyword arguments for the reverse function
        from the given url_kwargs and context_kwargs.

        :param view_name: The name of the view that you want.
        :param user: The user who is requesting the view
        :param url_kwargs: The url keyword arguments that came \
        with the request object. The view itself is responsible \
        to remove arguments that would not be part of a normal match \
        for that view. This is done by calling  the `get_url_kwargs` \
        method on the view.
        :param context_kwargs: Extra arguments that will be passed \
        to the view for consideration in the final keyword arguments \
        for reverse.
        :param follow_parent: If we encounter a parent reference should \
        we follow it. Defaults to True.
        :param check_permisions: Run permissions checks. Defaults to True.
        """

        view, url_name = self.get_initialized_view_and_name(view_name,
                                            follow_parent=follow_parent)

        if isinstance(view, URLAlias):
            view_name = view.get_view_name(view_name)
            bundle = view.get_bundle(self, url_kwargs, context_kwargs)

            if bundle and isinstance(bundle, Bundle):
                return bundle.get_view_url(view_name, user,
                                           url_kwargs=url_kwargs,
                                           context_kwargs=context_kwargs,
                                           follow_parent=follow_parent,
                                           check_permissions=check_permissions)

        elif view:

            # Get kwargs from view
            if not url_kwargs:
                url_kwargs = {}

            url_kwargs = view.get_url_kwargs(context_kwargs, **url_kwargs)
            view.kwargs = url_kwargs

            if check_permissions and not view.can_view(user):
                return None

            url = reverse("admin:%s" % url_name, kwargs=url_kwargs)
            return url
def get_object_header_view(self, request, url_kwargs, parent_only=False,
                                render_type='object_header'):
        """
        An object header is the title block of a CMS page. Actions
        to linked to in the header are based on this views
        bundle.

        This returns a view instance and view name of the view that
        should be rendered as an object header the view used is specified
        in `self.object_view`. If not match is found None, None is returned

        :param request: The request object
        :param url_kwargs: Any url keyword arguments as a dictionary
        :param parent_only: If `True` then the view will only \
        be rendered if object_view points to parent. This is usually \
        what you want to avoid extra lookups to get the object \
        you already have.
        :param render_type: The render type to use for the header. \
        Defaults to 'object_header'.
        """

        if parent_only and self.object_view != self.parent_attr:
            return None, None

        if self.object_view == self.parent_attr and self.parent:
            return self.parent.get_object_header_view(request, url_kwargs,
                                                    render_type=render_type)
        elif self.object_view:
            view, name = self.get_initialized_view_and_name(self.object_view,
                                    can_submit=False,
                                    base_template='cms/partial.html',
                                    request=request, kwargs=url_kwargs,
                                    render_type=render_type)
            if view and view.can_view(request.user):
                return view, name
        return None, None
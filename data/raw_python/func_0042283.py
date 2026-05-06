def update_kwargs(self, request, **kwargs):
        """
        Adds variables to the context that are expected by the
        base cms templates.

        * **navigation** - The side navigation for this bundle and user.
        * **dashboard** - The list of dashboard links for this user.
        * **object_header** - If no 'object_header' was passed in the \
        current context and the current bundle is set to get it's \
        object_header from it's parent, this will get that view and render \
        it as a string. Otherwise 'object_header will remain unset.
        * **subitem** - This is set to true if we rendered a new object_header \
        and the object used to render that string is not present in the \
        context args as 'obj'. This effects navigation and wording in the \
        templates.
        """

        kwargs = super(CMSRender, self).update_kwargs(request, **kwargs)

        # Check if we need to to include a separate object
        # bundle for the title
        bundle = kwargs.get('bundle')
        url_kwargs = kwargs.get('url_params')
        view = None
        if bundle:
            view, name = bundle.get_object_header_view(request, url_kwargs, parent_only=True)

        kwargs['dashboard'] = bundle.admin_site.get_dashboard_urls(request)

        if view:
            obj = view.get_object()
            if not 'object_header' in kwargs:
                kwargs['object_header'] = bundle._render_view_as_string(view, name, request, url_kwargs)
            if obj and obj != kwargs.get('obj'):
                kwargs['subitem'] = True
        return kwargs
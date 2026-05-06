def get_render_data(self, **kwargs):
        """
        Returns all data that should be passed to the renderer.
        By default adds the following arguments:

        * **bundle** - The bundle that is attached to this view instance.
        * **url_params** - The url keyword arguments. i.e.: self.kwargs.
        * **user** - The user attached to this request.
        * **base** - Unless base was already specified this gets set to \
        'self.base_template'.
        * **navigation** - The navigation bar for the page
        * **object_header_tmpl** - The template to use for the \
        object_header. Set to `self.object_header_tmpl`.
        * **back_bundle** - The back_back bundle is bundle that is linked to \
        from the object header as part of navigation. If there is an 'obj' \
        argument in the context to render, this will be set to the bundle \
        pointed to by the `main_list` attribute of this view's bundle. \
        If this is not set, the template's back link will point to the \
        admin_site's home page.
        """

        obj = getattr(self, 'object', None)
        data = dict(self.extra_render_data)
        data.update(kwargs)
        data.update({
            'bundle': self.bundle,
            'navigation': self.get_navigation(),
            'url_params': self.kwargs,
            'user': self.request.user,
            'object_header_tmpl': self.object_header_tmpl,
            'view_tags': tag_handler.tags_to_string(self.get_tags(obj))
        })

        if not 'base' in data:
            data['base'] = self.base_template

        if not 'back_bundle' in data:
            data['back_bundle'] = self.get_back_bundle()

        return super(CMSView, self).get_render_data(**data)
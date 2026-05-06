def get_admin_urls_for_registration(self):
        """
        Utilised by Wagtail's 'register_admin_urls' hook to register urls for
        our the views that class offers.
        """
        urls = (
            url(get_url_pattern(self.opts),
                self.index_view, name=get_url_name(self.opts)),
            url(get_url_pattern(self.opts, 'create'),
                self.create_view, name=get_url_name(self.opts, 'create')),
            url(get_object_specific_url_pattern(self.opts, 'edit'),
                self.edit_view, name=get_url_name(self.opts, 'edit')),
            url(get_object_specific_url_pattern(self.opts, 'confirm_delete'),
                self.confirm_delete_view,
                name=get_url_name(self.opts, 'confirm_delete')),
        )
        if self.inspect_view_enabled:
            urls = urls + (
                url(get_object_specific_url_pattern(self.opts, 'inspect'),
                    self.inspect_view,
                    name=get_url_name(self.opts, 'inspect')),
            )
        if self.is_pagemodel:
            urls = urls + (
                url(get_url_pattern(self.opts, 'choose_parent'),
                    self.choose_parent_view,
                    name=get_url_name(self.opts, 'choose_parent')),
                url(get_object_specific_url_pattern(self.opts, 'unpublish'),
                    self.unpublish_view,
                    name=get_url_name(self.opts, 'unpublish')),
                url(get_object_specific_url_pattern(self.opts, 'copy'),
                    self.copy_view,
                    name=get_url_name(self.opts, 'copy')),
            )
        return urls

        def construct_main_menu(self, request, menu_items):
            warnings.warn((
                "The 'construct_main_menu' method is now deprecated. You "
                "should also remove the construct_main_menu hook from "
                "wagtail_hooks.py in your app folder."), DeprecationWarning)
            return menu_items
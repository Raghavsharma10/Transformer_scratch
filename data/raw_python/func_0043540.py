def get_urls(self):
        """
        Overload the admin's urls for WYMEditor.
        """
        entry_admin_urls = super(EntryAdminWYMEditorMixin, self).get_urls()
        urls = [
            url(r'^wymeditor/$',
                self.admin_site.admin_view(self.wymeditor),
                name='zinnia_entry_wymeditor'),
        ]
        return urls + entry_admin_urls
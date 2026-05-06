def _media(self):
        """
        The medias needed to enhance the admin page.
        """
        def static_url(url):
            return staticfiles_storage.url('zinnia_wymeditor/%s' % url)

        media = super(EntryAdminWYMEditorMixin, self).media

        media += Media(
            js=(static_url('js/jquery.min.js'),
                static_url('js/wymeditor/jquery.wymeditor.pack.js'),
                static_url('js/wymeditor/plugins/hovertools/'
                           'jquery.wymeditor.hovertools.js'),
                reverse('admin:zinnia_entry_wymeditor')))
        return media
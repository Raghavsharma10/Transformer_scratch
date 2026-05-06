def process_request(self, request):
        """
        Reloads glitter URL patterns if page URLs change.

        Avoids having to restart the server to recreate the glitter URLs being used by Django.
        """
        global _urlconf_pages

        page_list = list(
            Page.objects.exclude(glitter_app_name='').values_list('id', 'url').order_by('id')
        )

        with _urlconf_lock:
            if page_list != _urlconf_pages:
                glitter_urls = 'glitter.urls'
                if glitter_urls in sys.modules:
                    importlib.reload(sys.modules[glitter_urls])
                _urlconf_pages = page_list
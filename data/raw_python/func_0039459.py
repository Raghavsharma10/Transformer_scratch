def _get_mirror_urls(self, mirrors=None, main_mirror_url=None):
        """Retrieves a list of URLs from the main mirror DNS entry
        unless a list of mirror URLs are passed.
        """
        if not mirrors:
            mirrors = get_mirrors(main_mirror_url)
            # Should this be made "less random"? E.g. netselect like?
            random.shuffle(mirrors)

        mirror_urls = set()
        for mirror_url in mirrors:
            # Make sure we have a valid URL
            if not ("http://" or "https://" or "file://") in mirror_url:
                mirror_url = "http://%s" % mirror_url
            if not mirror_url.endswith("/simple"):
                mirror_url = "%s/simple/" % mirror_url
            mirror_urls.add(mirror_url)

        return list(mirror_urls)
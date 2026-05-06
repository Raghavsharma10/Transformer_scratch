def get_short_url(self, obj):
        """
        Get short URL of blog post like '/blog/<slug>/' using ``get_absolute_url`` if available.
        Removes dependency on reverse URLs of Mezzanine views when deploying Mezzanine only as an API backend.
        """
        try:
            url = obj.get_absolute_url()
        except NoReverseMatch:
            url = '/blog/' + obj.slug
        return url
def save(self, force_insert=False, force_update=False, using=None, update_fields=None):
        """ Set html field with correct iframe. """
        if self.url:
            iframe_html = '<iframe src="{}" frameborder="0" title="{}" allowfullscreen></iframe>'
            self.html = iframe_html.format(
                self.get_embed_url(),
                self.title
            )
        return super().save(force_insert, force_update, using, update_fields)
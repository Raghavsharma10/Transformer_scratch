def derive_title(self):
        """
        Derives our title from our list
        """
        title = super(SmartListView, self).derive_title()

        if not title:
            return force_text(self.model._meta.verbose_name_plural).title()
        else:
            return title
def derive_title(self):
        """
        Derives our title from our object
        """
        if not self.title:
            return _("Create %s") % force_text(self.model._meta.verbose_name).title()
        else:
            return self.title
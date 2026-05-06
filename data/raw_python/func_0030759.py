def set_meta(self, instance):
        """
        Set django-meta stuff from LandingPageModel instance.
        """
        self.use_title_tag = True
        self.title = instance.title
def get_absolute_url(self):
        """produces a url to link directly to this instance, given the URL config

        :return: `str`
        """
        try:
            url = reverse("content-detail-view", kwargs={"pk": self.pk, "slug": self.slug})
        except NoReverseMatch:
            url = None
        return url
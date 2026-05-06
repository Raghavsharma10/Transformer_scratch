def get_checks_admin_reliability_warning_url(self):
        """
        When service Realiability is going down users should go to the
        the check history to find problem causes.
        :return: admin url with check list for this instance
        """
        # TODO: cache this.
        path = self.get_checks_admin_url()
        content_type = ContentType.objects.get_for_model(self)
        params = "?content_type__id__exact={0}&q={1}&success__exact=0".format(
            content_type.id,
            self.id
        )
        url = path + params
        return url
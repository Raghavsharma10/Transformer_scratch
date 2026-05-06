def get_download_urls(self):
        """
        Use like this:
            def get_urls(self):
                return self.get_download_urls() + super().get_urls()

        Returns: File download URLs for this model.
        """
        info = self.model._meta.app_label, self.model._meta.model_name
        return [
            url(r'^.+(' + self.upload_to + '/.+)/$', self.file_download_view, name='%s_%s_file_download' % info),
        ]
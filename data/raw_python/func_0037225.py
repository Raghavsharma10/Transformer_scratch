def get_upload_form(self):
        """Construct form for accepting file upload."""
        return self.form_class(self.request.POST, self.request.FILES)
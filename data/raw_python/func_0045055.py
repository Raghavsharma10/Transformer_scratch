def complete(self, request, response):
        """ Complete net auth.
        """
        extra = self.get_extra_data(response)
        data = {}
        for form_field, backend_field in self.PROFILE_MAPPING.items():
            data[form_field] = self.extract_data(extra, backend_field)
        request.session['extra'] = data

        if settings.ACCEPT_EXTRA_FORM:
            self.fill_extra_fields(request, data)

        request.session['identity'] = self.identity
        return redirect('netauth-extra', self.provider)
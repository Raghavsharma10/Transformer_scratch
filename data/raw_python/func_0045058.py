def fill_extra_fields(self, request, data):
        """
        Try to fetch extra data from provider, if this data is enough
        to validate settings.EXTRA_FORM then call save method of form
        class and login the user.

        The extra parameter can be some complex object
        this is why we use method function 'extra_data' to
        extract data from this object.

        Also we need to create a dictionary with remapped
        keys from profile mapping settings.
        """
        form = str_to_class(settings.EXTRA_FORM)(data)
        if form.is_valid():
            form.save(request, self.identity, self.provider)
            self.login_user(request)
        else:
            return data
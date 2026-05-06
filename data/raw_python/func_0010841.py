def get_success_url(self):
        """
        By default we use the referer that was stuffed in our
        form when it was created
        """
        if self.success_url:
            # if our smart url references an object, pass that in
            if self.success_url.find('@') > 0:
                return smart_url(self.success_url, self.object)
            else:
                return smart_url(self.success_url, None)

        elif 'loc' in self.form.cleaned_data:
            return self.form.cleaned_data['loc']

        raise ImproperlyConfigured("No redirect location found, override get_success_url to not use redirect urls")
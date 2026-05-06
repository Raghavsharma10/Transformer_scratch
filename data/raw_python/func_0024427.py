def _cache_form_details(self, form):
        """
        Caches some form details to lates process and validate incoming (response) form data

        Args:
            form: form dict
        """
        cache = FormCache()
        form['model']['form_key'] = cache.form_id
        form['model']['form_name'] = self.__class__.__name__
        cache.set(
            {
                'model': list(form['model'].keys()),  # In Python 3, dictionary keys are not serializable
                'non_data_fields': self.non_data_fields
            }
        )
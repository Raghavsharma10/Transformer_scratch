def get_factory_kwargs(self):
        """
        Let's us specify any extra parameters we might want to call for our form factory.

        These can include: 'form', 'fields', 'exclude' or 'formfield_callback'
        """
        params = dict()

        exclude = self.derive_exclude()
        exclude += self.derive_readonly()

        if self.fields:
            fields = list(self.fields)
            for ex in exclude:
                if ex in fields:
                    fields.remove(ex)

            params['fields'] = fields

        if exclude:
            params['exclude'] = exclude

        return params
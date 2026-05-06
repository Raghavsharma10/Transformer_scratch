def get_form(self):
        """
        Returns an instance of the form to be used in this view.
        """
        self.form = super(SmartFormMixin, self).get_form()

        fields = list(self.derive_fields())

        # apply our field filtering on our form class
        exclude = self.derive_exclude()
        exclude += self.derive_readonly()

        # remove any excluded fields
        for field in exclude:
            if field in self.form.fields:
                del self.form.fields[field]

        if fields is not None:
            # filter out our form fields
            remove = [name for name in self.form.fields.keys() if name not in fields]
            for name in remove:
                del self.form.fields[name]

        # stuff in our referer as the default location for where to return
        location = forms.CharField(widget=forms.widgets.HiddenInput(), required=False)

        if ('HTTP_REFERER' in self.request.META):
            location.initial = self.request.META['HTTP_REFERER']

        # add the location to our form fields
        self.form.fields['loc'] = location

        if fields:
            fields.append('loc')

        # provides a hook to programmatically customize fields before rendering
        for (name, field) in self.form.fields.items():
            field = self.customize_form_field(name, field)
            self.form.fields[name] = field

        return self.form
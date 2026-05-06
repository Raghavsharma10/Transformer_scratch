def render(self, name, value, attrs=None, renderer=None):
        """Include a hidden input to store the serialized upload value."""
        location = getattr(value, '_seralized_location', '')
        if location and not hasattr(value, 'url'):
            value.url = '#'
            if hasattr(self, 'get_template_substitution_values'):
                # Django 1.8-1.10
                self.template_with_initial = (
                    '%(initial_text)s: %(initial)s %(clear_template)s'
                    '<br />%(input_text)s: %(input)s')
        attrs = attrs or {}
        attrs.update({'data-upload-url': self.url})
        hidden_name = self.get_hidden_name(name)
        kwargs = {}
        if django_version >= (1, 11):
            kwargs['renderer'] = renderer

        parent = super(StickyUploadWidget, self).render(name, value, attrs=attrs, **kwargs)
        hidden = forms.HiddenInput().render(hidden_name, location, **kwargs)

        return mark_safe(parent + '\n' + hidden)
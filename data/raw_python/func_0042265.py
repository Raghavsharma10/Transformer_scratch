def formfield(self, form_class=None, choices_form_class=None, **kwargs):
        """
        Returns a django.forms.Field instance for this database Field.
        """
        defaults = {
            'required': not self.blank,
            'label': capfirst(self.verbose_name),
            'help_text': self.help_text,
        }
        if self.has_default():
            if callable(self.default):
                defaults['initial'] = self.default
                defaults['show_hidden_initial'] = True
            else:
                defaults['initial'] = self.get_default()

        include_blank = (self.blank
                         or not (self.has_default()
                                 or 'initial' in kwargs))

        choices = [BLANK_CHOICE_DASH, ] if include_blank else []
        choices.extend([
            (
                x.name,
                getattr(x, 'verbose_name', x.name) or x.name,
                getattr(x, 'help_text', None) or None
            )
            for x in self.choices_class.constants()
        ])

        defaults['choices'] = choices
        defaults['coerce'] = self.to_python

        if self.null:
            defaults['empty_value'] = None

        # Many of the subclass-specific formfield arguments (min_value,
        # max_value) don't apply for choice fields, so be sure to only pass
        # the values that TypedChoiceField will understand.
        for k in list(kwargs):
            if k not in ('coerce', 'empty_value', 'choices', 'required',
                         'widget', 'label', 'initial', 'help_text',
                         'error_messages', 'show_hidden_initial'):
                del kwargs[k]

        defaults.update(kwargs)
        form_class = choices_form_class or ChoicesFormField
        return form_class(**defaults)
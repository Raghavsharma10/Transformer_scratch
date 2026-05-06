def get_formset_form_class(self):
        """
        Returns the form class for use in the formset.

        If a form_class attribute or change_fields
        is provided then a form will be constructed
        with that. Otherwise None is returned.
        """
        if self.form_class or self.change_fields:
            params = {'formfield_callback': self.formfield_for_dbfield}
            if self.form_class:
                fc = self.customize_form_widgets(self.form_class)
                params['form'] = fc
            if self.change_fields:
                params['fields'] = self.change_fields

            return model_forms.modelform_factory(self.model, **params)
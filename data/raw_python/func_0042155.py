def get_formset_class(self, **kwargs):
        """
        Returns the formset for the queryset,
        if a form class is available.
        """
        form_class = self.get_formset_form_class()
        if form_class:
            kwargs['formfield_callback'] = self.formfield_for_dbfield
            return model_forms.modelformset_factory(self.model,
                        form_class, fields=self.change_fields, extra=0,
                        **kwargs)
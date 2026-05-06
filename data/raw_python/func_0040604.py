def update_form_labels(self, request=None, obj=None, form=None):
        """Returns a form obj after modifying form labels
        referred to in custom_form_labels.
        """
        for form_label in self.custom_form_labels:
            if form_label.field in form.base_fields:
                label = form_label.get_form_label(
                    request=request, obj=obj, model=self.model, form=form
                )
                if label:
                    form.base_fields[form_label.field].label = mark_safe(label)
        return form
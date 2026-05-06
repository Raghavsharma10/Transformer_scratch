def save_formsets(self, form, formsets, auto_tags=None):
        """
        Hook for saving formsets. Loops through
        all the given formsets and calls their
        save method.
        """
        for formset in formsets.values():
            tag_handler.set_auto_tags_for_formset(formset, auto_tags)
            formset.save()
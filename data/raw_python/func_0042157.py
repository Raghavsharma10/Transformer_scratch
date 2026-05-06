def get_visible_fields(self, formset):
        """
        Returns a list of visible fields. This
        are all the fields in `self.display_fields`
        plus any visible fields in the given formset
        minus any hidden fields in the formset.
        """

        visible_fields = list(self.display_fields)
        if formset:
            for x in formset.empty_form.visible_fields():
                if not x.name in visible_fields:
                    visible_fields.append(x.name)

            for x in formset.empty_form.hidden_fields():
                if x.name in visible_fields:
                    visible_fields.remove(x.name)

        return visible_fields
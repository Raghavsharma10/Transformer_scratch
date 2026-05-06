def previous_obj(self):
        """Returns a model obj that is the first occurrence of a previous
        obj relative to this object's appointment.

        Override this method if not am EDC subject model / CRF.
        """
        previous_obj = None
        if self.previous_visit:
            try:
                previous_obj = self.model.objects.get(
                    **{f"{self.model.visit_model_attr()}": self.previous_visit}
                )
            except ObjectDoesNotExist:
                pass
        return previous_obj
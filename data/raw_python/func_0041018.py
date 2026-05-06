def previous_visit(self):
        """Returns the previous visit for this request or None.

        Requires attr `visit_model_cls`.
        """
        previous_visit = None
        if self.appointment:
            appointment = self.appointment
            while appointment.previous_by_timepoint:
                try:
                    previous_visit = self.model.visit_model_cls().objects.get(
                        appointment=appointment.previous_by_timepoint
                    )
                except ObjectDoesNotExist:
                    pass
                else:
                    break
                appointment = appointment.previous_by_timepoint
        return previous_visit
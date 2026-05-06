def appointment(self):
        """Returns the appointment instance for this request or None.
        """
        return django_apps.get_model(self.appointment_model).objects.get(
            pk=self.request.GET.get("appointment")
        )
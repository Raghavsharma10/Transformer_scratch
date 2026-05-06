def sms_recipients(self):
        """Returns a list of recipients subscribed to receive SMS's
        for this "notifications" class.

        See also: edc_auth.UserProfile.
        """
        sms_recipients = []
        UserProfile = django_apps.get_model("edc_auth.UserProfile")
        for user_profile in UserProfile.objects.filter(
            user__is_active=True, user__is_staff=True
        ):
            try:
                user_profile.sms_notifications.get(name=self.name)
            except ObjectDoesNotExist:
                pass
            else:
                if user_profile.mobile:
                    sms_recipients.append(user_profile.mobile)
        return sms_recipients
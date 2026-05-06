def notify(
        self,
        force_notify=None,
        use_email=None,
        use_sms=None,
        email_body_template=None,
        **kwargs,
    ):
        """Notify / send an email and/or SMS.

        Main entry point.

        This notification class (me) knows from whom and to whom the
        notifications will be sent.

        See signals and kwargs are:
            * history_instance
            * instance
            * user
        """
        email_sent = None
        sms_sent = None
        use_email = use_email or getattr(settings, "EMAIL_ENABLED", False)
        use_sms = use_sms or getattr(settings, "TWILIO_ENABLED", False)
        if force_notify or self._notify_on_condition(**kwargs):
            if use_email:
                email_body_template = (
                    email_body_template or self.email_body_template
                ) + self.email_footer_template
                email_sent = self.send_email(
                    email_body_template=email_body_template, **kwargs
                )
            if use_sms:
                sms_sent = self.send_sms(**kwargs)
            self.post_notification_actions(
                email_sent=email_sent, sms_sent=sms_sent, **kwargs
            )
        return True if email_sent or sms_sent else False
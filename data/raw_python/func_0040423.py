def notify(self, force_notify=None, use_email=None, use_sms=None, **kwargs):
        """Overridden to only call `notify` if model matches.
        """
        notified = False
        instance = kwargs.get("instance")
        if instance._meta.label_lower == self.model:
            notified = super().notify(
                force_notify=force_notify,
                use_email=use_email,
                use_sms=use_sms,
                **kwargs,
            )
        return notified
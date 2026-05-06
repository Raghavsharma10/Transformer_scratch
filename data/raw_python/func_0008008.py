def send(self, *args, **kwargs):
        """
        Send email message, render if it is not rendered yet.

        Note
        ----
        Any extra arguments are passed to
        :class:`EmailMultiAlternatives.send() <django.core.mail.EmailMessage>`.

        Keyword Arguments
        -----------------
        clean : bool
            If ``True``, remove any template specific properties from the
            message object. Default is ``False``.
        """
        clean = kwargs.pop('clean', False)
        if not self._is_rendered:
            self.render()
        if clean:
            self.clean()
        return super(EmailMessage, self).send(*args, **kwargs)
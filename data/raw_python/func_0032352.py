def sendEmail(self, url, attempt, email, _sendEmail=_sendEmail):
        """
        Send an email for the given L{_PasswordResetAttempt}.

        @type url: L{URL}
        @param url: The URL of the password reset page.

        @type attempt: L{_PasswordResetAttempt}
        @param attempt: An L{Item} representing a particular user's attempt to
        reset their password.

        @type email: C{str}
        @param email: The email will be sent to this address.
        """

        host = url.netloc.split(':', 1)[0]
        from_ = 'reset@' + host

        body = file(sibpath(__file__, 'reset.rfc2822')).read()
        body %= {'from': from_,
                 'to': email,
                 'date': rfc822.formatdate(),
                 'message-id': smtp.messageid(),
                 'link': url.child(attempt.key)}

        _sendEmail(from_, email, body)
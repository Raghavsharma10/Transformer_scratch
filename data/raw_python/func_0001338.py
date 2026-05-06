def _login(self):
        '''Login to the SMTP server specified at instantiation

        Returns an authenticated SMTP instance.
        '''
        server, port, mode, debug = self.connection_details

        if mode == 'SSL':
            smtp_class = smtplib.SMTP_SSL
        else:
            smtp_class = smtplib.SMTP

        smtp = smtp_class(server, port)
        smtp.set_debuglevel(debug)

        if mode == 'TLS':
            smtp.starttls()

        self.authenticate(smtp)

        return smtp
def send_mail(self, event):
        """Connect to mail server and send actual email"""

        mime_mail = MIMEText(event.text)
        mime_mail['Subject'] = event.subject

        if event.account == 'default':
            account_name = self.config.default_account
        else:
            account_name = event.account

        account = list(filter(lambda account: account['name'] == account_name, self.config.accounts))[0]

        mime_mail['From'] = render(account['mail_from'], {'server': account['server'], 'hostname': self.hostname})
        mime_mail['To'] = event.to_address

        self.log('MimeMail:', mime_mail, lvl=verbose)
        if self.config.mail_send is True:
            self.log('Sending mail to', event.to_address)

            self.fireEvent(task(send_mail_worker, account, mime_mail, event), "mail-transmit-workers")
        else:
            self.log('Not sending mail, here it is for debugging info:', mime_mail, pretty=True)
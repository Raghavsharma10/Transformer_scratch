def get_one_mail(self):
        """
            Choose and return a random email from the mail archive.

        :return: Tuple containing From Address, To Address and the mail body.
        """

        while True:
            mail_key = random.choice(self.mailbox.keys())
            mail = self.mailbox[mail_key]
            from_addr = mail.get_from()
            to_addr = mail['To']
            mail_body = mail.get_payload()
            if not from_addr or not to_addr:
                continue
            return from_addr, to_addr, mail_body
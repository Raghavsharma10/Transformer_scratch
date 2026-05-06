def randomMails(self, count=1):
        """
        Return random e-mails.

        :rtype: list
        :returns: list of random e-mails
        """
        self.check_count(count)

        random_nicks = self.rn.random_nicks(count=count)
        random_domains = sample(self.dmails, count)

        return [
            nick.lower() + "@" + domain for nick, domain in zip(random_nicks,
                                                                random_domains)
        ]
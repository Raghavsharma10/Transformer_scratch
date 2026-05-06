def send_notification(self, title, message, typ=1, url=None, sender=None):
        """
        sends a message to user of this role's private mq exchange

        """
        self.user.send_notification(title=title, message=message, typ=typ, url=url,
                                    sender=sender)
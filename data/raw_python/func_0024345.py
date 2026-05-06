def send_notification(self, title, message, typ=1, url=None, sender=None):
        """
        sends message to users private mq exchange
        Args:
            title:
            message:
            sender:
            url:
            typ:
        """
        self.created_channels.channel.add_message(
            channel_key=self.prv_exchange,
            body=message,
            title=title,
            typ=typ,
            url=url,
            sender=sender,
            receiver=self
        )
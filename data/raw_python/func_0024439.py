def set_message(self, title, msg, typ, url=None):
        """
        Sets user notification message.

        Args:
            title: Msg. title
            msg:  Msg. text
            typ: Msg. type
            url: Additional URL (if exists)

        Returns:
            Message ID.
        """
        return self.user.send_notification(title=title,
                                           message=msg,
                                           typ=typ,
                                           url=url)
def msg_box(self, msg, title=None, typ='info'):
        """
        Create a message box

        :param str msg:
        :param str title:
        :param str typ: 'info', 'error', 'warning'
        """
        self.output['msgbox'] = {'type': typ, "title": title or msg[:20], "msg": msg}
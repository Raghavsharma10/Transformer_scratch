def get_blink_cookie(self, name):
        """Gets a blink cookie value"""
        value = self.get_cookie(name)

        if value != None:
            self.clear_cookie(name)
            return escape.url_unescape(value)
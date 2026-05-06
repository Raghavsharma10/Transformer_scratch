def set_blink(self, message, type="info"):
        """
        Sets the blink, a one-time transactional message that is shown on the
        next page load
        """
        self.set_cookie("blink_message", escape.url_escape(message), httponly=True)
        self.set_cookie("blink_type", escape.url_escape(type), httponly=True)
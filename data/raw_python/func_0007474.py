def _login(self, failed=False):
        """
        Login prompt
        """
        if failed:
            content = self.LOGIN_TEMPLATE.format(failed_message="Login failed")
        else:
            content = self.LOGIN_TEMPLATE.format(failed_message="")
        return "200 OK", content, {"Content-Type": "text/html"}
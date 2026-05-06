def cookies(self):
    """ Returns a list of all cookies in cookie string format. """
    return [line.strip()
            for line in self.conn.issue_command("GetCookies").split("\n")
            if line.strip()]
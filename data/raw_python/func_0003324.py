async def sessionstart(self):
        "Start session. Must start service.utils.session.Session to use this method"
        if not hasattr(self, 'session') or not self.session:
            self.session, setcookies = await call_api(self.container, 'session', 'start', {'cookies':self.rawcookie})
            for nc in setcookies:
                self.sent_cookies = [c for c in self.sent_cookies if c.key != nc.key]
                self.sent_cookies.append(nc)
async def sessiondestroy(self):
        """
        Destroy current session. The session object is discarded and can no longer be used in other requests.
        """
        if hasattr(self, 'session') and self.session:
            setcookies = await call_api(self.container, 'session', 'destroy', {'sessionid':self.session.id})
            self.session.unlock()
            del self.session
            for nc in setcookies:
                self.sent_cookies = [c for c in self.sent_cookies if c.key != nc.key]
                self.sent_cookies.append(nc)
async def createcsrf(self, csrfarg = '_csrf'):
        """
        Create a anti-CSRF token in the session
        """
        await self.sessionstart()
        if not csrfarg in self.session.vars:
            self.session.vars[csrfarg] = uuid.uuid4().hex
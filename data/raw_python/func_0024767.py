async def connect(self):
        """Connect to KLF 200."""
        PYVLXLOG.warning("Connecting to KLF 200.")
        await self.connection.connect()
        login = Login(pyvlx=self, password=self.config.password)
        await login.do_api_call()
        if not login.success:
            raise PyVLXException("Login to KLF 200 failed, check credentials")
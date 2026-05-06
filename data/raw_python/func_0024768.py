async def update_version(self):
        """Retrieve version and protocol version from API."""
        get_version = GetVersion(pyvlx=self)
        await get_version.do_api_call()
        if not get_version.success:
            raise PyVLXException("Unable to retrieve version")
        self.version = get_version.version
        get_protocol_version = GetProtocolVersion(pyvlx=self)
        await get_protocol_version.do_api_call()
        if not get_protocol_version.success:
            raise PyVLXException("Unable to retrieve protocol version")
        self.protocol_version = get_protocol_version.version
        PYVLXLOG.warning(
            "Connected to: %s, protocol version: %s",
            self.version, self.protocol_version)
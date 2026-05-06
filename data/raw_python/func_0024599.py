async def pulse(self):
        """Send get state request to API to keep the connection alive."""
        get_state = GetState(pyvlx=self.pyvlx)
        await get_state.do_api_call()
        if not get_state.success:
            raise PyVLXException("Unable to send get state.")
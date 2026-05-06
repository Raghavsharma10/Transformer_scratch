async def set_state(self, parameter):
        """Set switch to desired state."""
        command_send = CommandSend(pyvlx=self.pyvlx, node_id=self.node_id, parameter=parameter)
        await command_send.do_api_call()
        if not command_send.success:
            raise PyVLXException("Unable to send command")
        self.parameter = parameter
        await self.after_update()
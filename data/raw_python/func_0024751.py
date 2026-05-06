async def load(self):
        """Load scenes from KLF 200."""
        json_response = await self.pyvlx.interface.api_call('scenes', 'get')
        self.data_import(json_response)
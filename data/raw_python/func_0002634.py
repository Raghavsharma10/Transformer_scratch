async def status_by_state(self, state: str) -> dict:
        """Return the CDC status for the specified state."""
        data = await self.raw_cdc_data()

        try:
            info = next((v for k, v in data.items() if state in k))
        except StopIteration:
            return {}

        return adjust_status(info)
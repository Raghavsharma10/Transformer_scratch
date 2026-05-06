async def _raw_state_data(self) -> list:
        """Return a list of states."""
        data = await self._request('get', 'states')
        return [
            location for location in data
            if location['name'] != 'United States'
        ]
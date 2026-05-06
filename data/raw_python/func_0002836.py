async def _raw_user_report_data(self) -> list:
        """Return user report data (if accompanied by latitude/longitude)."""
        data = await self._request('get', 'map/markers')
        return [
            location for location in data
            if location['latitude'] and location['longitude']
        ]
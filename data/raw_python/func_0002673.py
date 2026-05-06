async def status_by_coordinates(
            self, latitude: float, longitude: float) -> dict:
        """Get symptom data for the location nearest to the user's lat/lon."""
        return await self.nearest_by_coordinates(latitude, longitude)
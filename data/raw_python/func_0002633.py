async def status_by_coordinates(
            self, latitude: float, longitude: float) -> dict:
        """Return the CDC status for the provided latitude/longitude."""
        cdc_data = await self.raw_cdc_data()
        nearest = await self.nearest_by_coordinates(latitude, longitude)
        return adjust_status(cdc_data[nearest['state']['name']])
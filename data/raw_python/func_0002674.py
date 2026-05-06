async def status_by_zip(self, zip_code: str) -> dict:
        """Get symptom data for the provided ZIP code."""
        try:
            location = next((
                d for d in await self.user_reports()
                if d['zip'] == zip_code))
        except StopIteration:
            return {}

        return await self.status_by_coordinates(
            float(location['latitude']), float(location['longitude']))
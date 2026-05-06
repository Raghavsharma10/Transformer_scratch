async def nearest_by_coordinates(
            self, latitude: float, longitude: float) -> dict:
        """Get the nearest report (with local and state info) to a lat/lon."""
        # Since user data is more granular than state or CDC data, find the
        # user report whose coordinates are closest to the provided
        # coordinates:
        nearest_user_report = get_nearest_by_coordinates(
            await self.user_reports(), 'latitude', 'longitude', latitude,
            longitude)

        try:
            # If the user report corresponds to a known state in
            # flunearyou.org's database, we can safely assume that's the
            # correct state:
            nearest_state = next((
                state for state in await self.state_data()
                if state['place_id'] == nearest_user_report['contained_by']))
        except StopIteration:
            # If a place ID doesn't exist (e.g., ZIP Code 98012 doesn't have a
            # place ID), calculate the nearest state by measuring the distance
            # from the provided latitude/longitude to flunearyou.org's
            # latitude/longitude that defines each state:
            nearest_state = get_nearest_by_coordinates(
                await self.state_data(), 'lat', 'lon', latitude, longitude)

        return {'local': nearest_user_report, 'state': nearest_state}
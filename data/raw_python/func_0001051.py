def _pass_list(self) -> List[Dict[str, Any]]:
        """Extract next stops along the journey."""
        stops: List[Dict[str, Any]] = []
        for stop in self.journey.PassList.BasicStop:
            index = stop.get("index")
            station = stop.Location.Station.HafasName.Text.text
            station_id = stop.Location.Station.ExternalId.text
            stops.append({"index": index, "stationId": station_id, "station": station})
        return stops
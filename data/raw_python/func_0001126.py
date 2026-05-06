def data(self) -> Dict[str, Any]:
        """Return travel data."""
        data: Dict[str, Any] = {}
        data["station"] = self.station
        data["stationId"] = self.station_id
        data["filter"] = self.products_filter

        journeys = []
        for j in sorted(self.journeys, key=lambda k: k.real_departure)[
            : self.max_journeys
        ]:
            journeys.append(
                {
                    "product": j.product,
                    "number": j.number,
                    "trainId": j.train_id,
                    "direction": j.direction,
                    "departure_time": j.real_departure_time,
                    "minutes": j.real_departure,
                    "delay": j.delay,
                    "stops": [s["station"] for s in j.stops],
                    "info": j.info,
                    "info_long": j.info_long,
                    "icon": j.icon,
                }
            )
        data["journeys"] = journeys
        return data
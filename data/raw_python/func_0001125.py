async def get_departures(
        self,
        station_id: str,
        direction_id: Optional[str] = None,
        max_journeys: int = 20,
        products: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Fetch data from rmv.de."""
        self.station_id: str = station_id
        self.direction_id: str = direction_id

        self.max_journeys: int = max_journeys

        self.products_filter: str = _product_filter(products or ALL_PRODUCTS)

        base_url: str = _base_url()
        params: Dict[str, Union[str, int]] = {
            "selectDate": "today",
            "time": "now",
            "input": self.station_id,
            "maxJourneys": self.max_journeys,
            "boardType": "dep",
            "productsFilter": self.products_filter,
            "disableEquivs": "discard_nearby",
            "output": "xml",
            "start": "yes",
        }
        if self.direction_id:
            params["dirInput"] = self.direction_id

        url = base_url + urllib.parse.urlencode(params)

        try:
            with async_timeout.timeout(self._timeout):
                async with self._session.get(url) as response:
                    _LOGGER.debug(f"Response from RMV API: {response.status}")
                    xml = await response.read()
                    _LOGGER.debug(xml)
        except (asyncio.TimeoutError, aiohttp.ClientError):
            _LOGGER.error("Can not load data from RMV API")
            raise RMVtransportApiConnectionError()

        # pylint: disable=I1101
        try:
            self.obj = objectify.fromstring(xml)
        except (TypeError, etree.XMLSyntaxError):
            _LOGGER.debug(f"Get from string: {xml[:100]}")
            print(f"Get from string: {xml}")
            raise RMVtransportError()

        try:
            self.now = self.current_time()
            self.station = self._station()
        except (TypeError, AttributeError):
            _LOGGER.debug(
                f"Time/Station TypeError or AttributeError {objectify.dump(self.obj)}"
            )
            raise RMVtransportError()

        self.journeys.clear()
        try:
            for journey in self.obj.SBRes.JourneyList.Journey:
                self.journeys.append(RMVJourney(journey, self.now))
        except AttributeError:
            _LOGGER.debug(f"Extract journeys: {objectify.dump(self.obj.SBRes)}")
            raise RMVtransportError()

        return self.data()
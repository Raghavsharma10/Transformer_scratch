def initialize(self, emt_id, emt_pass):
        """Manual initialization of the interface attributes.

        This is useful when the interface must be declare but initialized later
        on with parsed configuration values.

        Args:
            emt_id (str): ID given by the server upon registration
            emt_pass (str): Token given by the server upon registration
        """
        self._emt_id = emt_id
        self._emt_pass = emt_pass

        # Initialize modules
        self.bus = BusApi(self)
        self.geo = GeoApi(self)
        self.parking = ParkingApi(self)
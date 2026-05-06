def _prepare_wsdl_objects(self):
        """
        Create the data structure and get it ready for the WSDL request.
        """
        self.CarrierCode = 'FDXE'
        self.RoutingCode = 'FDSD'
        self.Address = self.client.factory.create('Address')
        self.ShipDateTime = datetime.datetime.now().isoformat()
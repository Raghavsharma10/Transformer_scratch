def _prepare_wsdl_objects(self):
        """
        Create the data structure and get it ready for the WSDL request.
        """
        self.CarrierCode = 'FDXE'
        self.Origin = self.client.factory.create('Address')
        self.Destination = self.client.factory.create('Address')
        self.ShipDate = datetime.date.today().isoformat()
        self.Service = None
        self.Packaging = 'YOUR_PACKAGING'
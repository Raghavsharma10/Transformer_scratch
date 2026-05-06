def _prepare_wsdl_objects(self):
        """
        This is the data that will be used to create your shipment. Create
        the data structure and get it ready for the WSDL request.
        """

        # Default behavior is to not request transit information
        self.ReturnTransitAndCommit = False

        # This is the primary data structure for processShipment requests.
        self.RequestedShipment = self.client.factory.create('RequestedShipment')
        self.RequestedShipment.ShipTimestamp = datetime.datetime.now()

        # Defaults for TotalWeight wsdl object.
        total_weight = self.client.factory.create('Weight')
        # Start at nothing.
        total_weight.Value = 0.0
        # Default to pounds.
        total_weight.Units = 'LB'
        # This is the total weight of the entire shipment. Shipments may
        # contain more than one package.
        self.RequestedShipment.TotalWeight = total_weight

        # This is the top level data structure for Shipper information.
        shipper = self.client.factory.create('Party')
        shipper.Address = self.client.factory.create('Address')
        shipper.Contact = self.client.factory.create('Contact')

        # Link the ShipperParty to our master data structure.
        self.RequestedShipment.Shipper = shipper

        # This is the top level data structure for Recipient information.
        recipient_party = self.client.factory.create('Party')
        recipient_party.Contact = self.client.factory.create('Contact')
        recipient_party.Address = self.client.factory.create('Address')
        # Link the RecipientParty object to our master data structure.
        self.RequestedShipment.Recipient = recipient_party

        # Make sender responsible for payment by default.
        self.RequestedShipment.ShippingChargesPayment = self.create_wsdl_object_of_type('Payment')
        self.RequestedShipment.ShippingChargesPayment.PaymentType = 'SENDER'

        # Start with no packages, user must add them.
        self.RequestedShipment.PackageCount = 0
        self.RequestedShipment.RequestedPackageLineItems = []

        # This is good to review if you'd like to see what the data structure
        # looks like.
        self.logger.debug(self.RequestedShipment)
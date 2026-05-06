def _prepare_wsdl_objects(self):
        """
        This is the data that will be used to create your shipment. Create
        the data structure and get it ready for the WSDL request.
        """

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

        # This is the top level data structure Shipper Party information.
        shipper_party = self.client.factory.create('Party')
        shipper_party.Address = self.client.factory.create('Address')
        shipper_party.Contact = self.client.factory.create('Contact')

        # Link the Shipper Party to our master data structure.
        self.RequestedShipment.Shipper = shipper_party

        # This is the top level data structure for RecipientParty information.
        recipient_party = self.client.factory.create('Party')
        recipient_party.Contact = self.client.factory.create('Contact')
        recipient_party.Address = self.client.factory.create('Address')

        # Link the RecipientParty object to our master data structure.
        self.RequestedShipment.Recipient = recipient_party

        payor = self.client.factory.create('Payor')
        # Grab the account number from the FedexConfig object by default.
        # Assume US.
        payor.ResponsibleParty = self.client.factory.create('Party')
        payor.ResponsibleParty.Address = self.client.factory.create('Address')
        payor.ResponsibleParty.Address.CountryCode = 'US'

        # ShippingChargesPayment WSDL object default values.
        shipping_charges_payment = self.client.factory.create('Payment')
        shipping_charges_payment.Payor = payor
        shipping_charges_payment.PaymentType = 'SENDER'
        self.RequestedShipment.ShippingChargesPayment = shipping_charges_payment

        self.RequestedShipment.LabelSpecification = self.client.factory.create('LabelSpecification')

        # NONE, PREFERRED or LIST
        self.RequestedShipment.RateRequestTypes = ['PREFERRED']

        # Start with no packages, user must add them.
        self.RequestedShipment.PackageCount = 0
        self.RequestedShipment.RequestedPackageLineItems = []

        # This is good to review if you'd like to see what the data structure
        # looks like.
        self.logger.debug(self.RequestedShipment)
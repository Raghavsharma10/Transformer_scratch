def _prepare_wsdl_objects(self):
        """
        Preps the WSDL data structures for the user.
        """

        self.DeletionControlType = self.client.factory.create('DeletionControlType')
        self.TrackingId = self.client.factory.create('TrackingId')
        self.TrackingId.TrackingIdType = self.client.factory.create('TrackingIdType')
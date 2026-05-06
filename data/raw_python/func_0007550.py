def _assemble_and_send_request(self):
        """
        Fires off the Fedex request.
        
        @warning: NEVER CALL THIS METHOD DIRECTLY. CALL send_request(), WHICH RESIDES
            ON FedexBaseService AND IS INHERITED.
        """

        client = self.client
        # Fire off the query.
        return client.service.deleteShipment(
                WebAuthenticationDetail=self.WebAuthenticationDetail,
                ClientDetail=self.ClientDetail,
                TransactionDetail=self.TransactionDetail,
                Version=self.VersionId,
                ShipTimestamp=datetime.datetime.now(),
                TrackingId=self.TrackingId,
                DeletionControl=self.DeletionControlType)
def _assemble_and_send_validation_request(self):
        """
        Fires off the Fedex shipment validation request.
        
        @warning: NEVER CALL THIS METHOD DIRECTLY. CALL 
            send_validation_request(), WHICH RESIDES ON FedexBaseService 
            AND IS INHERITED.
        """

        # Fire off the query.
        return self.client.service.validateShipment(
                WebAuthenticationDetail=self.WebAuthenticationDetail,
                ClientDetail=self.ClientDetail,
                TransactionDetail=self.TransactionDetail,
                Version=self.VersionId,
                RequestedShipment=self.RequestedShipment)
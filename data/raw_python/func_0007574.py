def _assemble_and_send_request(self):
        """
        Fires off the Fedex request.
        
        @warning: NEVER CALL THIS METHOD DIRECTLY. CALL send_request(), 
            WHICH RESIDES ON FedexBaseService AND IS INHERITED.
        """

        # We get an exception like this when specifying an IntegratorId:
        # suds.TypeNotFound: Type not found: 'IntegratorId'
        # Setting it to None does not seem to appease it.
        del self.ClientDetail.IntegratorId
        self.logger.debug(self.WebAuthenticationDetail)
        self.logger.debug(self.ClientDetail)
        self.logger.debug(self.TransactionDetail)
        self.logger.debug(self.VersionId)
        # Fire off the query.
        return self.client.service.serviceAvailability(
                WebAuthenticationDetail=self.WebAuthenticationDetail,
                ClientDetail=self.ClientDetail,
                TransactionDetail=self.TransactionDetail,
                Version=self.VersionId,
                Origin=self.Origin,
                Destination=self.Destination,
                ShipDate=self.ShipDate,
                CarrierCode=self.CarrierCode,
                Service=self.Service,
                Packaging=self.Packaging)
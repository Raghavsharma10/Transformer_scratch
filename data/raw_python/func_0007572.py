def _assemble_and_send_request(self):
        """
        Fires off the Fedex request.
        
        @warning: NEVER CALL THIS METHOD DIRECTLY. CALL send_request(), WHICH RESIDES
            ON FedexBaseService AND IS INHERITED.
        """

        client = self.client

        # We get an exception like this when specifying an IntegratorId:
        # suds.TypeNotFound: Type not found: 'IntegratorId'
        # Setting it to None does not seem to appease it.

        del self.ClientDetail.IntegratorId

        # Fire off the query.
        response = client.service.postalCodeInquiry(WebAuthenticationDetail=self.WebAuthenticationDetail,
                                                    ClientDetail=self.ClientDetail,
                                                    TransactionDetail=self.TransactionDetail,
                                                    Version=self.VersionId,
                                                    PostalCode=self.PostalCode,
                                                    CountryCode=self.CountryCode,
                                                    CarrierCode=self.CarrierCode)

        return response
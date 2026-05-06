def _assemble_and_send_request(self):
        """
        Fires off the Fedex request.

        @warning: NEVER CALL THIS METHOD DIRECTLY. CALL send_request(),
            WHICH RESIDES ON FedexBaseService AND IS INHERITED.
        """

        # Fire off the query.
        return self.client.service.getPickupAvailability(
            WebAuthenticationDetail=self.WebAuthenticationDetail,
            ClientDetail=self.ClientDetail,
            TransactionDetail=self.TransactionDetail,
            Version=self.VersionId,
            PickupType=self.PickupType,
            AccountNumber=self.AccountNumber,
            PickupAddress=self.PickupAddress,
            PickupRequestType=self.PickupRequestType,
            DispatchDate=self.DispatchDate,
            NumberOfBusinessDays=self.NumberOfBusinessDays,
            PackageReadyTime=self.PackageReadyTime,
            CustomerCloseTime=self.CustomerCloseTime,
            Carriers=self.Carriers,
            ShipmentAttributes=self.ShipmentAttributes,
            PackageDetails=self.PackageDetails
        )
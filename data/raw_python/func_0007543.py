def _assemble_and_send_request(self):
        """
        Fires off the Fedex request.

        @warning: NEVER CALL THIS METHOD DIRECTLY. CALL send_request(),
            WHICH RESIDES ON FedexBaseService AND IS INHERITED.
        """

        # Fire off the query.
        return self.client.service.createPickup(
            WebAuthenticationDetail=self.WebAuthenticationDetail,
            ClientDetail=self.ClientDetail,
            TransactionDetail=self.TransactionDetail,
            Version=self.VersionId,
            OriginDetail=self.OriginDetail,
            PickupServiceCategory=self.PickupServiceCategory,
            PackageCount=self.PackageCount,
            TotalWeight=self.TotalWeight,
            CarrierCode=self.CarrierCode,
            OversizePackageCount=self.OversizePackageCount,
            Remarks=self.Remarks,
            CommodityDescription=self.CommodityDescription,
            CountryRelationship=self.CountryRelationship
        )
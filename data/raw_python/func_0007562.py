def _prepare_wsdl_objects(self):
        """
        This sets the package identifier information. This may be a tracking
        number or a few different things as per the Fedex spec.
        """

        self.SelectionDetails = self.client.factory.create('TrackSelectionDetail')

        # Default to Fedex
        self.SelectionDetails.CarrierCode = 'FDXE'

        track_package_id = self.client.factory.create('TrackPackageIdentifier')

        # Default to tracking number.
        track_package_id.Type = 'TRACKING_NUMBER_OR_DOORTAG'

        self.SelectionDetails.PackageIdentifier = track_package_id
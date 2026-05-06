def print_label(self, package_num=None):
        """
        Prints all of a shipment's labels, or optionally just one.
        
        @type package_num: L{int}
        @param package_num: 0-based index of the package to print. This is
                            only useful for shipments with more than one package.
        """

        if package_num:
            packages = [
                self.shipment.response.CompletedShipmentDetail.CompletedPackageDetails[package_num]
            ]
        else:
            packages = self.shipment.response.CompletedShipmentDetail.CompletedPackageDetails

        for package in packages:
            label_binary = binascii.a2b_base64(package.Label.Parts[0].Image)
            self._print_base64(label_binary)
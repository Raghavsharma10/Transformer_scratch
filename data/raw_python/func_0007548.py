def add_package(self, package_item):
        """
        Adds a package to the ship request.
        
        @type package_item: WSDL object, type of RequestedPackageLineItem 
            WSDL object.
        @keyword package_item: A RequestedPackageLineItem, created by
            calling create_wsdl_object_of_type('RequestedPackageLineItem') on
            this ShipmentRequest object. See examples/create_shipment.py for
            more details.
        """

        self.RequestedShipment.RequestedPackageLineItems.append(package_item)
        package_weight = package_item.Weight.Value
        self.RequestedShipment.TotalWeight.Value += package_weight
        self.RequestedShipment.PackageCount += 1
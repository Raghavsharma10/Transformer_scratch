def _prepare_wsdl_objects(self):
            """
            This is the data that will be used to create your shipment. Create
            the data structure and get it ready for the WSDL request.
            """
            self.UploadDocumentsRequest = self.client.factory.create('UploadDocumentsRequest')
            self.OriginCountryCode  =None
            self.DestinationCountryCode  =None
            self.Usage  ='ELECTRONIC_TRADE_DOCUMENTS'#Default Usage
            self.Documents = []
            self.UploadDocumentsRequest.Documents = []
            self.logger.debug(self.UploadDocumentsRequest)
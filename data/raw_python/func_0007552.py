def _assemble_and_send_request(self):
            """
            Fires off the Fedex request.

            @warning: NEVER CALL THIS METHOD DIRECTLY. CALL send_request(),
                WHICH RESIDES ON FedexBaseService AND IS INHERITED.
            """

            # Fire off the query.
            return self.client.service.uploadDocuments(
                    WebAuthenticationDetail=self.WebAuthenticationDetail,
                    ClientDetail=self.ClientDetail,
                    TransactionDetail=self.TransactionDetail,
                    Version=self.VersionId,
                    Documents=self.Documents,
                    Usage = self.Usage,
                    OriginCountryCode = self.OriginCountryCode,
                    DestinationCountryCode = self.DestinationCountryCode,
                )
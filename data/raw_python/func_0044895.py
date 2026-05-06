def parse_results(self, data):
        """parse_results

        Parses the MelissaData response.

        Args:
                data (dict): Contains MelissaData response

        Returns:
                results, either contains a dict with corrected address info or -1 for an invalid address.
        """
        results = []
        if len(data["Records"]) < 1:
            return -1

        codes = data["Records"][0]["Results"]
        for code in codes.split(","):
            results.append(str(code))

        self.addr1 = data["Records"][0]["AddressLine1"]
        self.addr2 = data["Records"][0]["AddressLine2"]
        self.city = data["Records"][0]["City"]
        self.name = data["Records"][0]["NameFull"]
        self.phone = data["Records"][0]["PhoneNumber"]
        self.province = data["Records"][0]["State"]
        self.postal = data["Records"][0]["PostalCode"]
        self.recordID = data["Records"][0]["RecordID"]
        return results
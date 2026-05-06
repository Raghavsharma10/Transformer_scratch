def verify_address(self, addr1="", addr2="", city="", fname="", lname="", phone="", province="", postal="", country="", email="", recordID="", freeform= ""):
        """verify_address

        Builds a JSON request to send to Melissa data. Takes in all needed address info.

        Args:
                addr1 (str):Contains info for Melissa data
                addr2 (str):Contains info for Melissa data
                city (str):Contains info for Melissa data
                fname (str):Contains info for Melissa data
                lname (str):Contains info for Melissa data
                phone (str):Contains info for Melissa data
                province (str):Contains info for Melissa data
                postal (str):Contains info for Melissa data
                country (str):Contains info for Melissa data
                email (str):Contains info for Melissa data
                recordID (str):Contains info for Melissa data
                freeform (str):Contains info for Melissa data

        Returns:
            result, a string containing the result codes from MelissaData
        """
        data = {
            "TransmissionReference": "",
            "CustomerID": self.custID,
            "Actions": "Check",
            "Options": "",
            "Columns": "",
            "Records": [{
                "RecordID": recordID,
                "CompanyName": "",
                "FullName": fname + " " + lname,
                "AddressLine1": addr1,
                "AddressLine2": addr2,
                "Suite": "",
                "City": city,
                "State": province,
                "PostalCode": postal,
                "Country": country,
                "PhoneNumber": phone,
                "EmailAddress": email,
                "FreeForm": freeform,
            }]
        }
        self.country = country
        data = json.dumps(data)
        result = requests.post("https://personator.melissadata.net/v3/WEB/ContactVerify/doContactVerify", data=data)
        result = json.loads(result.text)
        result = self.parse_results(result)
        return result
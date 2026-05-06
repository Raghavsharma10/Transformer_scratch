def data(self):
        """Parameters passed to the API containing the details to create a new
         alert.

        :return: parameters to create new alert.
        :rtype: dict
        """
        data = {}
        data["name"] = self.name
        data["query"] = self.queryd
        data["languages"] = self.languages
        data["countries"] = self.countries if self.countries else ""
        data["sources"] = self.sources if self.sources else ""
        data["blocked_sites"] = self.blocked_sites if self.blocked_sites else ""
        data["noise_detection"] = self.noise_detection if self.noise_detection else ""
        data["reviews_pages"] = self.reviews_pages if self.reviews_pages else ""

        # Deletes parameter if it does not have a value
        for key, value in list(data.items()):
            if value == '':
                del data[key]

        data = json.dumps(data)
        return data
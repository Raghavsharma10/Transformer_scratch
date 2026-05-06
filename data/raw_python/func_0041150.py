def data(self):
        """Parameters passed to the API containing the details to update a
         alert.

        :return: parameters to create new alert.
        :rtype: dict
        """
        data = {}
        data["favorite"] = self.favorite if self.favorite else ""
        data["trashed"] = self.trashed if self.trashed else ""
        data["read"] = self.read if self.read else ""
        data["tags"] = self.tags if self.tags else ""
        data["folder"] = self.folder if self.folder else ""
        data["tone"] = self.tone if self.tone else ""

        # Deletes parameter if it does not have a value
        for key, value in list(data.items()):
            if value == '':
                del data[key]

        data = json.dumps(data)
        return data
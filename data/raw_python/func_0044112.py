def addCity(self, fileName):
        """Add a JSON file and read the users.

        :param fileName: path to the JSON file. This file has to have a list of
        users, called users.
        :type fileName: str.
        """
        with open(fileName) as data_file:
            data = load(data_file)
        for u in data["users"]:
            if not any(d["name"] == u["name"] for d in self.__users):
                self.__users.append(u)
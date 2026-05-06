def people(self):
        """
        Retrieve all people of the company

        :return: list of people objects
        :rtype: list
        """
        return fields.ListField(name=HightonConstants.PEOPLE, init_class=Person).decode(
            self.element_from_string(
                self._get_request(
                    endpoint=self.ENDPOINT + '/' + str(self.id) + '/people',
                ).text
            )
        )
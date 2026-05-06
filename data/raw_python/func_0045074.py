def update(self):
        """
        Updates the object

        :return:
        :rtype: response
        """
        return self._put_request(
            data=self.element_to_string(
                self.encode()
            ),
            endpoint=self.ENDPOINT + '/' + str(self.id)
        )
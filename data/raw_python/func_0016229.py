def set_user_attribute(self, username, attribute, value, raise_on_error=False):
        """Set an attribute on  a user
        :param username: The username on which to set the attribute
        :param attribute: The name of the attribute to set
        :param value: The value of the attribute to set
        :return: True on success, False on failure.
        """
        data = {
            'attributes': [
                {
                    'name': attribute,
                    'values': [
                        value
                    ]
                },
            ]
        }
        response = self._post(self.rest_url + "/user/attribute",
                              params={"username": username,},
                              data=json.dumps(data))

        if response.status_code == 204:
            return True

        if raise_on_error:
            raise RuntimeError(response.json()['message'])

        return False
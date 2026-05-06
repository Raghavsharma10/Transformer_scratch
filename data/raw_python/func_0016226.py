def add_user(self, username, raise_on_error=False, **kwargs):
        """Add a user to the directory

        Args:
            username: The account username
            raise_on_error: optional (default: False)
            **kwargs: key-value pairs:
                          password: mandatory
                          email: mandatory
                          first_name: optional
                          last_name: optional
                          display_name: optional
                          active: optional (default True)

        Returns:
            True: Succeeded
            False: If unsuccessful
        """
        # Check that mandatory elements have been provided
        if 'password' not in kwargs:
            raise ValueError("missing password")
        if 'email' not in kwargs:
            raise ValueError("missing email")

        # Populate data with default and mandatory values.
        # A KeyError means a mandatory value was not provided,
        # so raise a ValueError indicating bad args.
        try:
            data = {
                    "name": username,
                    "first-name": username,
                    "last-name": username,
                    "display-name": username,
                    "email": kwargs["email"],
                    "password": {"value": kwargs["password"]},
                    "active": True
                   }
        except KeyError:
            return ValueError

        # Remove special case 'password'
        del(kwargs["password"])

        # Put values from kwargs into data
        for k, v in kwargs.items():
            new_k = k.replace("_", "-")
            if new_k not in data:
                raise ValueError("invalid argument %s" % k)
            data[new_k] = v

        response = self._post(self.rest_url + "/user",
                              data=json.dumps(data))

        if response.status_code == 201:
            return True

        if raise_on_error:
            raise RuntimeError(response.json()['message'])

        return False
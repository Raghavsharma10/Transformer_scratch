def _convert_to_identifier_json(self, address_data):
        """Convert input address data into json format"""

        if isinstance(address_data, str):
            # allow just passing a slug string.
            return {"slug": address_data}

        if isinstance(address_data, tuple) and len(address_data) > 0:
            address_json = {"address": address_data[0]}
            if len(address_data) > 1:
                address_json["zipcode"] = address_data[1]
            if len(address_data) > 2:
                address_json["meta"] = address_data[2]
            return address_json

        if isinstance(address_data, dict):
            allowed_keys = ["address", "zipcode", "unit", "city", "state", "slug", "meta",
                            "client_value", "client_value_sqft"]

            # ensure the dict does not contain any unallowed keys
            for key in address_data:
                if key not in allowed_keys:
                    msg = "Key in address input not allowed: " + key
                    raise housecanary.exceptions.InvalidInputException(msg)

            # ensure it contains an "address" key
            if "address" in address_data or "slug" in address_data:
                return address_data

        # if we made it here, the input was not valid.
        msg = ("Input is invalid. Must be a list of (address, zipcode) tuples, or a dict or list"
               " of dicts with each item containing at least an 'address' or 'slug' key.")
        raise housecanary.exceptions.InvalidInputException((msg))
def check_is_uuid(self, uuid_str: str):
        """Check if it's an Isogeo UUID handling specific form.

        :param str uuid_str: UUID string to check
        """
        # check uuid type
        if not isinstance(uuid_str, str):
            raise TypeError("'uuid_str' expected a str value.")
        else:
            pass
        # handle Isogeo specific UUID in XML exports
        if "isogeo:metadata" in uuid_str:
            uuid_str = "urn:uuid:{}".format(uuid_str.split(":")[-1])
        else:
            pass
        # test it
        try:
            uid = UUID(uuid_str)
            return uid.hex == uuid_str.replace("-", "").replace("urn:uuid:", "")
        except ValueError as e:
            logging.error(
                "uuid ValueError. {} ({})  -- {}".format(type(uuid_str), uuid_str, e)
            )
            return False
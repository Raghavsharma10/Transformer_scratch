def convert_uuid(self, in_uuid: str = str, mode: bool = 0):
        """Convert a metadata UUID to its URI equivalent. And conversely.

        :param str in_uuid: UUID or URI to convert
        :param int mode: conversion direction. Options:

          * 0 to HEX
          * 1 to URN (RFC4122)
          * 2 to URN (Isogeo specific style)

        """
        # parameters check
        if not isinstance(in_uuid, str):
            raise TypeError("'in_uuid' expected a str value.")
        else:
            pass
        if not checker.check_is_uuid(in_uuid):
            raise ValueError("{} is not a correct UUID".format(in_uuid))
        else:
            pass
        if not isinstance(mode, int):
            raise TypeError("'mode' expects an integer value")
        else:
            pass
        # handle Isogeo specific UUID in XML exports
        if "isogeo:metadata" in in_uuid:
            in_uuid = "urn:uuid:{}".format(in_uuid.split(":")[-1])
            logging.debug("Isogeo UUUID URN spotted: {}".format(in_uuid))
        else:
            pass
        # operate
        if mode == 0:
            return uuid.UUID(in_uuid).hex
        elif mode == 1:
            return uuid.UUID(in_uuid).urn
        elif mode == 2:
            urn = uuid.UUID(in_uuid).urn
            return "urn:isogeo:metadata:uuid:{}".format(urn.split(":")[2])
        else:
            raise ValueError("'mode' must be  one of: 0 | 1 | 2")
def set(self, oid, value, value_type=None):
        """
        Sets a single OID value. If you do not pass value_type hnmp will
        try to guess the correct type. Autodetection is supported for:

        * int and float (as Integer, fractional part will be discarded)
        * IPv4 address (as IpAddress)
        * str (as OctetString)

        Unfortunately, pysnmp does not support the SNMP FLOAT type so
        please use Integer instead.
        """
        snmpsecurity = self._get_snmp_security()

        if value_type is None:
            if isinstance(value, int):
                data = Integer(value)
            elif isinstance(value, float):
                data = Integer(value)
            elif isinstance(value, str):
                if is_ipv4_address(value):
                    data = IpAddress(value)
                else:
                    data = OctetString(value)
            else:
                raise TypeError(
                    "Unable to autodetect type. Please pass one of "
                    "these strings as the value_type keyword arg: "
                    ", ".join(TYPES.keys())
                )
        else:
            if value_type not in TYPES:
                raise ValueError("'{}' is not one of the supported types: {}".format(
                    value_type,
                    ", ".join(TYPES.keys())
                ))
            data = TYPES[value_type](value)

        try:
            engine_error, pdu_error, pdu_error_index, objects = self._cmdgen.setCmd(
                snmpsecurity,
                cmdgen.UdpTransportTarget((self.host, self.port), timeout=self.timeout,
                                          retries=self.retries),
                (oid, data),
            )
            if engine_error:
                raise SNMPError(engine_error)
            if pdu_error:
                raise SNMPError(pdu_error.prettyPrint())
        except Exception as e:
            raise SNMPError(e)

        _, value = objects[0]
        value = _convert_value_to_native(value)
        return value
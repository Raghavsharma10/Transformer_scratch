def get(self, oid):
        """
        Get a single OID value.
        """
        snmpsecurity = self._get_snmp_security()

        try:
            engine_error, pdu_error, pdu_error_index, objects = self._cmdgen.getCmd(
                snmpsecurity,
                cmdgen.UdpTransportTarget((self.host, self.port), timeout=self.timeout,
                                          retries=self.retries),
                oid,
            )

        except Exception as e:
            raise SNMPError(e)
        if engine_error:
            raise SNMPError(engine_error)
        if pdu_error:
            raise SNMPError(pdu_error.prettyPrint())

        _, value = objects[0]
        value = _convert_value_to_native(value)
        return value
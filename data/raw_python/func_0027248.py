def _prepare_abi(self, jsonabi):
        """
        Prepare the contract json abi for sighash lookups and fast access

        :param jsonabi: contracts abi in json format
        :return:
        """
        self.signatures = {}
        for element_description in jsonabi:
            abi_e = AbiMethod(element_description)
            if abi_e["type"] == "constructor":
                self.signatures[b"__constructor__"] = abi_e
            elif abi_e["type"] == "fallback":
                abi_e.setdefault("inputs", [])
                self.signatures[b"__fallback__"] = abi_e
            elif abi_e["type"] == "function":
                # function and signature present
                # todo: we could generate the sighash ourselves? requires keccak256
                if abi_e.get("signature"):
                    self.signatures[Utils.str_to_bytes(abi_e["signature"])] = abi_e
            elif abi_e["type"] == "event":
                self.signatures[b"__event__"] = abi_e
            else:
                raise Exception("Invalid abi type: %s - %s - %s" % (abi_e.get("type"),
                                                                    element_description, abi_e))
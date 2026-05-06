def from_input_lookup(s):
        """
        Return a new AbiMethod object from an input stream
        :param s: binary input
        :return: new AbiMethod object matching the provided input stream
        """
        for pseudo_abi in FourByteDirectory.get_pseudo_abi_for_input(s):
            method = AbiMethod(pseudo_abi)
            types_def = pseudo_abi["inputs"]
            types = [t["type"] for t in types_def]
            names = [t["name"] for t in types_def]

            values = decode_abi(types, s[4:])

            # (type, name, data)
            method.inputs = [{"type": t, "name": n, "data": v} for t, n, v in list(
                zip(types, names, values))]
            return method
def describe_constructor(self, s):
        """
        Describe the input bytesequence (constructor arguments) s based on the loaded contract
         abi definition

        :param s: bytes constructor arguments
        :return: AbiMethod instance
        """
        method = self.signatures.get(b"__constructor__")
        if not method:
            # constructor not available
            m = AbiMethod({"type": "constructor", "name": "", "inputs": [], "outputs": []})
            return m

        types_def = method["inputs"]
        types = [t["type"] for t in types_def]
        names = [t["name"] for t in types_def]

        if not len(s):
            values = len(types) * ["<nA>"]
        else:
            values = decode_abi(types, s)

        # (type, name, data)
        method.inputs = [{"type": t, "name": n, "data": v} for t, n, v in list(
            zip(types, names, values))]
        return method
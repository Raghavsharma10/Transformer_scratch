def describe_input(self, s):
        """
        Describe the input bytesequence s based on the loaded contract abi definition

        :param s: bytes input
        :return: AbiMethod instance
        """
        signatures = self.signatures.items()

        for sighash, method in signatures:
            if sighash is None or sighash.startswith(b"__"):
                continue  # skip constructor

            if s.startswith(sighash):
                s = s[len(sighash):]

                types_def = self.signatures.get(sighash)["inputs"]
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
        else:
            method = AbiMethod({"type": "fallback",
                                "name": "__fallback__",
                                "inputs": [], "outputs": []})
            types_def = self.signatures.get(b"__fallback__", {"inputs": []})["inputs"]
            types = [t["type"] for t in types_def]
            names = [t["name"] for t in types_def]

            values = decode_abi(types, s)

            # (type, name, data)
            method.inputs = [{"type": t, "name": n, "data": v} for t, n, v in list(
                zip(types, names, values))]
            return method
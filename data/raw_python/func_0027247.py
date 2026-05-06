def get_pseudo_abi_for_input(s, timeout=None, proxies=None):
        """
        Lookup sighash from 4bytes.directory, create a pseudo api and try to decode it with the parsed abi.
        May return multiple results as sighashes may collide.
        :param s: bytes input
        :return: pseudo abi for method
        """
        sighash = Utils.bytes_to_str(s[:4])
        for pseudo_abi in FourByteDirectory.get_pseudo_abi_for_sighash(sighash, timeout=timeout, proxies=proxies):
            types = [ti["type"] for ti in pseudo_abi['inputs']]
            try:
                # test decoding
                _ = decode_abi(types, s[4:])
                yield pseudo_abi
            except eth_abi.exceptions.DecodingError as e:
                continue
def _pfp__set_packer(self, pack_type, packer=None, pack=None, unpack=None, func_call_info=None):
        """Set the packer/pack/unpack functions for this field, as
        well as the pack type.

        :pack_type: The data type of the packed data
        :packer: A function that can handle packing and unpacking. First
                 arg is true/false (to pack or unpack). Second arg is the stream.
                 Must return an array of chars.
        :pack: A function that packs data. It must accept an array of chars and return an
                array of chars that is a packed form of the input.
        :unpack: A function that unpacks data. It must accept an array of chars and
                return an array of chars
        """
        self._pfp__pack_type = pack_type
        self._pfp__unpack = unpack
        self._pfp__pack = pack
        self._pfp__packer = packer
        self._pfp__pack_func_call_info = func_call_info
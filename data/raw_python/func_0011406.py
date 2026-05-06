def _pfp__unpack_data(self, raw_data):
        """Means that the field has already been parsed normally,
        and that it now needs to be unpacked.

        :raw_data: A string of the data that the field consumed while parsing
        """
        if self._pfp__pack_type is None:
            return
        if self._pfp__no_unpack:
            return

        unpack_func = self._pfp__packer
        unpack_args = []
        if self._pfp__packer is not None:
            unpack_func = self._pfp__packer
            unpack_args = [false(), raw_data]

        elif self._pfp__unpack is not None:
            unpack_func = self._pfp__unpack
            unpack_args = [raw_data]

        # does not need to be converted to a char array
        if not isinstance(unpack_func, functions.NativeFunction):
            io_stream = bitwrap.BitwrappedStream(six.BytesIO(raw_data))
            unpack_args[-1] = Array(len(raw_data), Char, io_stream)

        res = unpack_func.call(unpack_args, *self._pfp__pack_func_call_info, no_cast=True)
        if isinstance(res, Array):
            res = res._pfp__build()

        io_stream = six.BytesIO(res)
        tmp_stream = bitwrap.BitwrappedStream(io_stream)

        tmp_stream.padded = self._pfp__interp.get_bitfield_padded()

        self._ = self._pfp__parsed_packed = self._pfp__pack_type(tmp_stream)

        self._._pfp__watch(self)
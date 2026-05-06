def _pfp__pack_data(self):
        """Pack the nested field
        """
        if self._pfp__pack_type is None:
            return

        tmp_stream = six.BytesIO()
        self._._pfp__build(bitwrap.BitwrappedStream(tmp_stream))
        raw_data = tmp_stream.getvalue()

        unpack_func = self._pfp__packer
        unpack_args = []
        if self._pfp__packer is not None:
            unpack_func = self._pfp__packer
            unpack_args = [true(), raw_data]
        elif self._pfp__pack is not None:
            unpack_func = self._pfp__pack
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

        self._pfp__no_unpack = True
        self._pfp__parse(tmp_stream)
        self._pfp__no_unpack = False
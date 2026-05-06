def serialize(self, value, compress_level=-1):
        """
        Serializes a value based on its type.

        :param value: Something to be serialized
        :type value: six.string_types, int, long, object
        :param compress_level: How much to compress.
            0 = no compression, 1 = fastest, 9 = slowest but best,
            -1 = default compression level.
        :type compress_level: int
        :return: Serialized type
        :rtype: str
        """
        flags = 0
        if isinstance(value, binary_type):
            flags |= self.FLAGS['binary']
        elif isinstance(value, text_type):
            value = value.encode('utf8')
        elif isinstance(value, int) and isinstance(value, bool) is False:
            flags |= self.FLAGS['integer']
            value = str(value)
        elif isinstance(value, long) and isinstance(value, bool) is False:
            flags |= self.FLAGS['long']
            value = str(value)
        else:
            flags |= self.FLAGS['object']
            buf = BytesIO()
            pickler = self.pickler(buf, self.pickle_protocol)
            pickler.dump(value)
            value = buf.getvalue()

        if compress_level != 0 and len(value) > self.COMPRESSION_THRESHOLD:
            if compress_level is not None and compress_level > 0:
                # Use the specified compression level.
                compressed_value = self.compression.compress(value, compress_level)
            else:
                # Use the default compression level.
                compressed_value = self.compression.compress(value)
            # Use the compressed value only if it is actually smaller.
            if compressed_value and len(compressed_value) < len(value):
                value = compressed_value
                flags |= self.FLAGS['compressed']

        return flags, value
def from_byte_array(cls, bytes_):
        """Decodes a run-length encoded ByteArray and returns a Bitmap.
        The ByteArray decompresses to a sequence of 32-bit values, which are
        stored as a byte string. (The specific encoding depends on Form.depth.)
        """
        runs = cls._length_run_coding.parse(bytes_)
        pixels = (run.pixels for run in runs.data)
        data = "".join(itertools.chain.from_iterable(pixels))
        return cls(data)
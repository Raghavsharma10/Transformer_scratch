def save(self, filename):
        """Save a project in .lsdsng format to the target file.

        :param filename: the name of the file to which to save

        :deprecated: use ``save_lsdsng(filename)`` instead
        """
        with open(filename, 'wb') as fp:
            writer = BlockWriter()
            factory = BlockFactory()

            preamble_dummy_bytes = bytearray([0] * 9)
            preamble = bread.parse(
                preamble_dummy_bytes, spec.lsdsng_preamble)
            preamble.name = self.name
            preamble.version = self.version

            preamble_data = bread.write(preamble)
            raw_data = self.get_raw_data()
            compressed_data = filepack.compress(raw_data)

            writer.write(compressed_data, factory)

            fp.write(preamble_data)

            for key in sorted(factory.blocks.keys()):
                fp.write(bytearray(factory.blocks[key].data))
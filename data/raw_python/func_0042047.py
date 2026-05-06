def copy_data(self):
        """
        Copy the data from the it's point of origin, serializing it,
        storing it serialized as well as in it's raw form and calculate
        a running hash of the serialized representation
        """
        HASH_FUNCTION = hashlib.sha256()

        try:
            raw_iterator = self.get_binary_iterator()
        except AttributeError:
            raw_iterator = self.get_non_binary_iterator()
            self.copy_file = tempfile.NamedTemporaryFile(mode='w+')

            for part in raw_iterator:
                encoded_part = dbsafe_encode(part)
                self.copy_file.write(encoded_part)
                self.copy_file.write('\n')
                HASH_FUNCTION.update(encoded_part)

            self.copy_file.seek(0)
            self.data_iterator = (dbsafe_decode(line) for line in self.copy_file)

        else:
            self.copy_file = tempfile.NamedTemporaryFile(mode='w+b')

            for part in raw_iterator:
                self.copy_file.write(part)
                HASH_FUNCTION.update(part)

            self.copy_file.seek(0)
            self.data_iterator = self.copy_file

        self.new_hash = HASH_FUNCTION.hexdigest()
def save_seedhex_file(self, path: str) -> None:
        """
        Save hexadecimal seed file from seed

        :param path: Authentication file path
        """
        seedhex = convert_seed_to_seedhex(self.seed)
        with open(path, 'w') as fh:
            fh.write(seedhex)
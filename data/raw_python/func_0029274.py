def set_terminal_key(self, encrypted_key):
        """
        Change the terminal key. The encrypted_key is a hex string.
        encrypted_key is expected to be encrypted under master key
        """
        if encrypted_key:
            try:
                new_key = bytes.fromhex(encrypted_key)
                if len(self.terminal_key) != len(new_key):
                    # The keys must have equal length
                    return False

                self.terminal_key = self.tmk_cipher.decrypt(new_key)
                self.store_terminal_key(raw2str(self.terminal_key))

                self.tpk_cipher = DES3.new(self.terminal_key, DES3.MODE_ECB)
                self.print_keys()
                return True

            except ValueError:
                return False

        return False
def get_encrypted_pin(self, clear_pin, card_number):
        """
        Get PIN block in ISO 0 format, encrypted with the terminal key
        """
        if not self.terminal_key:
            print('Terminal key is not set')
            return ''

        if self.pinblock_format == '01':
            try:
                pinblock = bytes.fromhex(get_pinblock(clear_pin, card_number))
                #print('PIN block: {}'.format(raw2str(pinblock)))
            except TypeError:
                return ''

            encrypted_pinblock = self.tpk_cipher.encrypt(pinblock)
            return raw2str(encrypted_pinblock)

        else:
            print('Unsupported PIN Block format')
            return ''
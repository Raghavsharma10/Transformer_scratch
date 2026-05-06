def crypt(self, data, crypt_type):
        """Crypt the data in blocks, running it through des_crypt()"""

        # Error check the data
        if not data:
            return ''
        if len(data) % self.block_size != 0:
            if crypt_type == des.DECRYPT: # Decryption must work on 8 byte blocks
                raise ValueError("Invalid data length, data must be a multiple of " + str(self.block_size) + " bytes\n" + data + ".")
            if not self.getPadding():
                raise ValueError("Invalid data length, data must be a multiple of " + str(self.block_size) + " bytes\n. Try setting the optional padding character")
            else:
                data += (self.block_size - (len(data) % self.block_size)) * self.getPadding()
            # print "Len of data: %f" % (len(data) / self.block_size)

        if self.getMode() == CBC:
            if self.getIV():
                iv = self.__String_to_BitList(self.getIV())
            else:
                raise ValueError("For CBC mode, you must supply the Initial Value (IV) for ciphering")

        # Split the data into blocks, crypting each one seperately
        i = 0
        result = []
        while i < len(data):
            block = self.__String_to_BitList(data[i:i+8])

            # Xor with IV if using CBC mode
            if self.getMode() == CBC:
                if crypt_type == des.ENCRYPT:
                    block = list(map(lambda x, y: x ^ y, block, iv))

                processed_block = self.__des_crypt(block, crypt_type)

                if crypt_type == des.DECRYPT:
                    processed_block = list(map(lambda x, y: x ^ y, processed_block, iv))
                    iv = block
                else:
                    iv = processed_block
            else:
                processed_block = self.__des_crypt(block, crypt_type)


            # Add the resulting crypted block to our list
            result.append(self.__BitList_to_String(processed_block))
            i += 8

        # Return the full crypted string
        if _pythonMajorVersion < 3:
            return ''.join(result)
        else:
            return bytes.fromhex('').join(result)
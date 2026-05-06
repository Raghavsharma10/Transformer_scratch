def pad(cls, data):
        """
        Pads data to match AES block size
        """
        if sys.version_info > (3, 0):
            try:
                data = data.encode("utf-8")
            except AttributeError:
                pass

            length = AES.block_size - (len(data) % AES.block_size)
            data += bytes([length]) * length
            return data
        else:
            return data + (AES.block_size - len(data) % AES.block_size) * chr(AES.block_size - len(data) % AES.block_size)
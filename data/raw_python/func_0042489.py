def unpad(cls, data):
        """
        Unpads data that has been padded
        """
        if sys.version_info > (3, 0):
            return data[:-ord(data[len(data)-1:])].decode()
        else:
            return data[:-ord(data[len(data)-1:])]
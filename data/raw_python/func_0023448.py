def pack(cls, data):
        '''Pack the provided data into a Response'''
        return struct.pack('>ll', len(data) + 4, cls.FRAME_TYPE) + data
def tileXYZToQuadKey(self, x, y, z):
        '''
        Computes quadKey value based on tile x, y and z values.
        '''
        quadKey = ''
        for i in range(z, 0, -1):
            digit = 0
            mask = 1 << (i - 1)
            if (x & mask) != 0:
                digit += 1
            if (y & mask) != 0:
                digit += 2
            quadKey += str(digit)
        return quadKey
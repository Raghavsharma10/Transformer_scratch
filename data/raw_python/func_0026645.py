def quadKeyToTileXYZ(self, quadKey):
        '''
        Computes tile x, y and z values based on quadKey.
        '''
        tileX = 0
        tileY = 0
        tileZ = len(quadKey)

        for i in range(tileZ, 0, -1):
            mask = 1 << (i - 1)
            value = quadKey[tileZ - i]

            if value == '0':
                continue

            elif value == '1':
                tileX |= mask

            elif value == '2':
                tileY |= mask

            elif value == '3':
                tileX |= mask
                tileY |= mask

            else:
                raise Exception('Invalid QuadKey')

        return (tileX, tileY, tileZ)
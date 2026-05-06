def createTileUrl(self, x, y, z):
        '''
        returns new tile url based on template
        '''
        return self.tileTemplate.replace('{{x}}', str(x)).replace('{{y}}', str(
            y)).replace('{{z}}', str(z))
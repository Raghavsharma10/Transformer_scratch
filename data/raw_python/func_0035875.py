def isTransiting(self):
        """ Checks the the istransiting tag to see if the planet transits. Note that this only works as of catalogue
        version  ee12343381ae4106fd2db908e25ffc537a2ee98c (11th March 2014) where the istransiting tag was implemented
        """
        try:
            isTransiting = self.params['istransiting']
        except KeyError:
            return False

        if isTransiting == '1':
            return True
        else:
            return False
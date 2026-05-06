def writetofile(self, filename):
        '''Writes the in-memory zip to a file.'''
        f = open(filename, "w")
        f.write(self.read())
        f.close()
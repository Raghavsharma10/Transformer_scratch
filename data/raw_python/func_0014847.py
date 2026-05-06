def tofile(self, filename, format = 'ascii'):
        """Save VTK data to file.
        """
        if not common.is_string(filename):
            raise TypeError('argument filename must be string but got %s'%(type(filename)))
        if format not in ['ascii','binary']:
            raise TypeError('argument format must be ascii | binary')
        filename = filename.strip()
        if not filename:
            raise ValueError('filename must be non-empty string')
        if filename[-4:]!='.vtk':
            filename += '.vtk'
        f = open(filename,'wb')
        f.write(self.to_string(format))
        f.close()
def getfullfilename(file_path):
        ''' Get full filename (with extension)
        '''
        warnings.warn("getfullfilename() is deprecated and will be removed in near future. Use chirptext.io.write_file() instead", DeprecationWarning)
        if file_path:
            return os.path.basename(file_path)
        else:
            return ''
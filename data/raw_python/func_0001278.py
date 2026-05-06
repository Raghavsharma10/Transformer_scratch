def locate_config(self):
        ''' Locate config file '''
        for f in self.__potential:
            f = FileHelper.abspath(f)
            if os.path.isfile(f):
                return f
        return None
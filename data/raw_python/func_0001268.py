def content(self):
        ''' Return report content as a string if mode == STRINGIO else an empty string '''
        if isinstance(self.__report_file, io.StringIO):
            return self.__report_file.getvalue()
        else:
            return ''
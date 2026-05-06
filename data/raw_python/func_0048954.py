def pretty(self):
        '''
        Return a string like '/foo/bar.py:230 in foo.bar.my_func'.
        '''
        return '{}:{} in {}.{}'.format(
            self.filename,
            self.line_number,
            self.module_name,
            self.function_name)
def replace(self, str1, str2):
        """ Set verbatim code replacement
        
        It is strongly recommended to use function['$foo'] = 'bar' where
        possible because template variables are less likely to changed
        than the code itself in future versions of vispy.
        
        Parameters
        ----------
        str1 : str
            String to replace
        str2 : str
            String to replace str1 with
        """
        if str2 != self._replacements.get(str1, None):
            self._replacements[str1] = str2
            self.changed(code_changed=True)
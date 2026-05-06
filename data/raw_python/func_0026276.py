def _build_parmlist(self, parameters):
        """
        Converts a dictionary of name and value pairs into a 
        PARMLIST string value acceptable to the Payflow Pro API.

        """

        args = []
        for key, value in parameters.items():
            if not value is None:
                # We always use the explicit-length keyname format, to reduce the chance
                # of requests failing due to unusual characters in parameter values.

                try:
                    classinfo = unicode
                except NameError:
                    classinfo = str

                if isinstance(value, classinfo):
                    key = '%s[%d]' % (key.upper(), len(value.encode('utf-8')))
                else:
                    key = '%s[%d]' % (key.upper(), len(str(value)))
                args.append('%s=%s' % (key, value))
        args.sort()
        parmlist = '&'.join(args)        
        return parmlist
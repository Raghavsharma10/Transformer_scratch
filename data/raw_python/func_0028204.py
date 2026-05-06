def join(self, joiner, formatter=lambda s, t: t.format(s),
             template="{}"):
        """Join values and convert to string

        Example:

            >>> from ww import l
            >>> lst = l('012')
            >>> lst.join(',')
            u'0,1,2'
            >>> lst.join(',', template="{}#")
            u'0#,1#,2#'
            >>> string = lst.join(',',\
                                  formatter = lambda x, y: str(int(x) ** 2))
            >>> string
            u'0,1,4'
        """

        return ww.s(joiner).join(self, formatter, template)
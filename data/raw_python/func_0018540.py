def parse(self, text, **kwargs):
        '''Parse the given text and return result from MeCab.

        :param text: the text to parse.
        :type text: str
        :param as_nodes: return generator of MeCabNodes if True;
            or string if False.
        :type as_nodes: bool, defaults to False
        :param boundary_constraints: regular expression for morpheme boundary
            splitting; if non-None and feature_constraints is None, then
            boundary constraint parsing will be used.
        :type boundary_constraints: str or re
        :param feature_constraints: tuple containing tuple instances of
            target morpheme and corresponding feature string in order
            of precedence; if non-None and boundary_constraints is None,
            then feature constraint parsing will be used.
        :type feature_constraints: tuple
        :return: A single string containing the entire MeCab output;
            or a Generator yielding the MeCabNode instances.
        :raises: MeCabError
        '''
        if text is None:
            logger.error(self._ERROR_EMPTY_STR)
            raise MeCabError(self._ERROR_EMPTY_STR)
        elif not isinstance(text, str):
            logger.error(self._ERROR_NOTSTR)
            raise MeCabError(self._ERROR_NOTSTR)
        elif 'partial' in self.options and not text.endswith("\n"):
            logger.error(self._ERROR_MISSING_NL)
            raise MeCabError(self._ERROR_MISSING_NL)

        if self._KW_BOUNDARY in kwargs:
            val = kwargs[self._KW_BOUNDARY]
            if not isinstance(val, self._REGEXTYPE) and not isinstance(val, str):
                logger.error(self._ERROR_BOUNDARY)
                raise MeCabError(self._ERROR_BOUNDARY)
        elif self._KW_FEATURE in kwargs:
            val = kwargs[self._KW_FEATURE]
            if not isinstance(val, tuple):
                logger.error(self._ERROR_FEATURE)
                raise MeCabError(self._ERROR_FEATURE)

        as_nodes = kwargs.get(self._KW_ASNODES, False)

        if as_nodes:
            return self.__parse_tonodes(text, **kwargs)
        else:
            return self.__parse_tostr(text, **kwargs)
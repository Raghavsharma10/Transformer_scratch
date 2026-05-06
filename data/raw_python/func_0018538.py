def __parse_tostr(self, text, **kwargs):
        '''Builds and returns the MeCab function for parsing Unicode text.

        Args:
            fn_name: MeCab function name that determines the function
                behavior, either 'mecab_sparse_tostr' or
                'mecab_nbest_sparse_tostr'.

        Returns:
            A function definition, tailored to parsing Unicode text and
            returning the result as a string suitable for display on stdout,
            using either the default or N-best behavior.
        '''
        n = self.options.get('nbest', 1)

        if self._KW_BOUNDARY in kwargs:
            patt = kwargs.get(self._KW_BOUNDARY, '.')
            tokens = list(self.__split_pattern(text, patt))
            text = ''.join([t[0] for t in tokens])

            btext = self.__str2bytes(text)
            self.__mecab.mecab_lattice_set_sentence(self.lattice, btext)

            bpos = 0
            self.__mecab.mecab_lattice_set_boundary_constraint(
                self.lattice, bpos, self.MECAB_TOKEN_BOUNDARY)

            for (token, match) in tokens:
                bpos += 1
                if match:
                    mark = self.MECAB_INSIDE_TOKEN
                else:
                    mark = self.MECAB_ANY_BOUNDARY

                for _ in range(1, len(self.__str2bytes(token))):
                    self.__mecab.mecab_lattice_set_boundary_constraint(
                        self.lattice, bpos, mark)
                    bpos += 1
                self.__mecab.mecab_lattice_set_boundary_constraint(
                    self.lattice, bpos, self.MECAB_TOKEN_BOUNDARY)
        elif self._KW_FEATURE in kwargs:
            features = kwargs.get(self._KW_FEATURE, ())
            fd = {morph: self.__str2bytes(feat) for morph, feat in features}

            tokens = self.__split_features(text, [e[0] for e in features])
            text = ''.join([t[0] for t in tokens])

            btext = self.__str2bytes(text)
            self.__mecab.mecab_lattice_set_sentence(self.lattice, btext)

            bpos = 0
            for chunk, match in tokens:
                c = len(self.__str2bytes(chunk))
                if match == True:
                    self.__mecab.mecab_lattice_set_feature_constraint(
                        self.lattice, bpos, bpos+c, fd[chunk])
                bpos += c
        else:
            btext = self.__str2bytes(text)
            self.__mecab.mecab_lattice_set_sentence(self.lattice, btext)

        self.__mecab.mecab_parse_lattice(self.tagger, self.lattice)

        if n > 1:
            res = self.__mecab.mecab_lattice_nbest_tostr(self.lattice, n)
        else:
            res = self.__mecab.mecab_lattice_tostr(self.lattice)

        if res != self.__ffi.NULL:
            raw = self.__ffi.string(res)
            return self.__bytes2str(raw).strip()
        else:
            err = self.__mecab.mecab_lattice_strerror(self.lattice)
            logger.error(self.__bytes2str(self.__ffi.string(err)))
            raise MeCabError(self.__bytes2str(self.__ffi.string(err)))
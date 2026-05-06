def __parse_tonodes(self, text, **kwargs):
        '''Builds and returns the MeCab function for parsing to nodes using
        morpheme boundary constraints.

        Args:
            format_feature: flag indicating whether or not to format the feature
                value for each node yielded.

        Returns:
            A function which returns a Generator, tailored to using boundary
            constraints and parsing as nodes, using either the default or
            N-best behavior.
        '''
        n = self.options.get('nbest', 1)

        try:
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
                    if match:
                        self.__mecab.mecab_lattice_set_feature_constraint(
                            self.lattice, bpos, bpos+c, fd[chunk])
                    bpos += c
            else:
                btext = self.__str2bytes(text)
                self.__mecab.mecab_lattice_set_sentence(self.lattice, btext)

            self.__mecab.mecab_parse_lattice(self.tagger, self.lattice)

            for _ in range(n):
                check = self.__mecab.mecab_lattice_next(self.lattice)
                if n == 1 or check:
                    nptr = self.__mecab.mecab_lattice_get_bos_node(self.lattice)
                    while nptr != self.__ffi.NULL:
                        # skip over any BOS nodes, since mecab does
                        if nptr.stat != MeCabNode.BOS_NODE:
                            raws = self.__ffi.string(
                                nptr.surface[0:nptr.length])
                            surf = self.__bytes2str(raws).strip()

                            if 'output_format_type' in self.options or \
                               'node_format' in self.options:
                                sp = self.__mecab.mecab_format_node(
                                    self.tagger, nptr)
                                if sp != self.__ffi.NULL:
                                    rawf = self.__ffi.string(sp)
                                else:
                                    err = self.__mecab.mecab_strerror(
                                            self.tagger)
                                    err = self.__bytes2str(
                                            self.__ffi.string(err))
                                    msg = self._ERROR_NODEFORMAT.format(
                                            surf, err)
                                    raise MeCabError(msg)
                            else:
                                rawf = self.__ffi.string(nptr.feature)
                            feat = self.__bytes2str(rawf).strip()

                            mnode = MeCabNode(nptr, surf, feat)
                            yield mnode
                        nptr = getattr(nptr, 'next')
        except GeneratorExit:
            logger.debug('close invoked on generator')
        except MeCabError:
            raise
        except:
            err = self.__mecab.mecab_lattice_strerror(self.lattice)
            logger.error(self.__bytes2str(self.__ffi.string(err)))
            raise MeCabError(self.__bytes2str(self.__ffi.string(err)))
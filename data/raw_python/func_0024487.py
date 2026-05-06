def _convert_reaction(self, reaction):
        if not (reaction['reactants'] or reaction['products'] or reaction['reagents']):
            raise ValueError('empty reaction')
        maps = {'reactants': [], 'products': [], 'reagents': []}
        for i, tmp in maps.items():
            for molecule in reaction[i]:
                used = set()
                for atom in molecule['atoms']:
                    m = atom['mapping']
                    if m:
                        if m in used:
                            if not self._ignore:
                                raise MappingError('mapping in molecules should be unique')
                            warning(f'non-unique mapping in molecule: {m}')
                        else:
                            used.add(m)
                    tmp.append(m)

        length = count(max(max(maps['products'], default=0), max(maps['reactants'], default=0),
                           max(maps['reagents'], default=0)) + 1)

        ''' map unmapped atoms.
        '''
        for i, tmp in maps.items():
            used = set()
            maps[i] = remap = []
            for m in tmp:
                if not m:
                    remap.append(next(length))
                elif m in used:
                    if not self._ignore:
                        raise MappingError('mapping in reagents or products or reactants should be unique')
                    # force remap non unique atoms in molecules.
                    remap.append(next(length))
                    warning(f'mapping changed: {m} to {remap[-1]}')
                else:
                    remap.append(m)
                    used.add(m)

        if maps['reagents']:
            tmp = (set(maps['reactants']) | set(maps['products'])) & set(maps['reagents'])
            if tmp:
                e = f'reagents has map intersection with reactants or products: {tmp}'
                if not self._ignore:
                    raise MappingError(e)
                warning(e)
                maps['reagents'] = [x if x not in tmp else next(length) for x in maps['reagents']]

        ''' find breaks in map. e.g. 1,2,5,6. 3,4 - skipped
        '''
        if self.__remap:
            lose = sorted(set(range(1, next(length))) - set(maps['reactants']) - set(maps['products']) -
                          set(maps['reagents']), reverse=True)
            if lose:
                for i, tmp in maps.items():
                    if not tmp:
                        continue
                    for j in lose:
                        maps[i] = tmp = [x if x < j else x - 1 for x in tmp]
        ''' end
        '''
        rc = ReactionContainer(meta=reaction['meta'])
        for i, tmp in maps.items():
            shift = 0
            for j in reaction[i]:
                atom_len = len(j['atoms'])
                remapped = {x: y for x, y in enumerate(tmp[shift: atom_len + shift])}
                shift += atom_len
                g = self.__convert_structure(j, remapped)
                rc[i].append(g)
        return rc
def _modifyInternal(self, *, sort=None, purge=False, done=None):
        """Creates a whole new database from existing one, based on given
        modifiers.

        :sort: pattern should look like this:
        ([(<index>, True|False)], {<level_index>: [(<index>, True|False)]}),
        where True|False indicate whether to reverse or not,
        <index> are one of Model.indexes and <level_index> indicate
        a number of level to sort.
        Of course, the lists above may contain multiple items.

        :done: patterns looks similar to :sort:, except that it has additional
        <regexp> values and that True|False means to mark as done|undone.

        @note: Should not be used directly. It was defined here, because
        :save: decorator needs undecorated version of Model.modify.

        :sort: Pattern on which to sort the database.
        :purge: Whether to purge done items.
        :done: Pattern on which to mark items as done/undone.
        :returns: New database, modified according to supplied arguments.

        """
        sortAll, sortLevels = sort is not None and sort or ([], {})
        doneAll, doneLevels = done is not None and done or ([], {})

        def _mark(v, i):
            if done is None:
                return v[:4]

            def _mark_(index, regexp, du):
                if du is None:
                    return v[:4]
                if index is None:
                    for v_ in v[:3]:
                        if regexp is None or re.match(regexp, str(v_)):
                            return v[:3] + [du]
                    return v[:4]
                if regexp is None or re.match(regexp, str(v[index])):
                    return v[:3] + [du]
            try:
                for doneLevel in doneLevels[i]:
                    result = _mark_(*doneLevel)
                if result is not None:
                    return result
            except KeyError:
                pass
            for doneAll_ in doneAll:
                result = _mark_(*doneAll_)
            if result is None:
                return v[:4]
            return result

        def _modify(submodel, i):
            _new = list()
            for v in submodel:
                if purge:
                    if not v[3]:
                        _new.append(_mark(v, i) + [_modify(v[4], i + 1)])
                else:
                    _new.append(_mark(v, i) + [_modify(v[4], i + 1)])
            levels = sortLevels.get(i) or sortLevels.get(str(i))
            for index, reverse in levels or sortAll:
                _new = sorted(_new, key=lambda e: e[index], reverse=reverse)
            return _new
        return _modify(self.data, 1)
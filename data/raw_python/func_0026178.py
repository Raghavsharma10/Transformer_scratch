def walk(self, function, raise_errors=True,
            call_on_sections=False, **keywargs):
        """
        Walk every member and call a function on the keyword and value.

        Return a dictionary of the return values

        If the function raises an exception, raise the errror
        unless ``raise_errors=False``, in which case set the return value to
        ``False``.

        Any unrecognised keyword arguments you pass to walk, will be pased on
        to the function you pass in.

        Note: if ``call_on_sections`` is ``True`` then - on encountering a
        subsection, *first* the function is called for the *whole* subsection,
        and then recurses into it's members. This means your function must be
        able to handle strings, dictionaries and lists. This allows you
        to change the key of subsections as well as for ordinary members. The
        return value when called on the whole subsection has to be discarded.

        See  the encode and decode methods for examples, including functions.

        admonition:: caution

        You can use ``walk`` to transform the names of members of a section
        but you mustn't add or delete members.

        >>> config = '''[XXXXsection]
        ... XXXXkey = XXXXvalue'''.splitlines()
        >>> cfg = ConfigObj(config)
        >>> cfg
        ConfigObj({'XXXXsection': {'XXXXkey': 'XXXXvalue'}})
        >>> def transform(section, key):
        ...     val = section[key]
        ...     newkey = key.replace('XXXX', 'CLIENT1')
        ...     section.rename(key, newkey)
        ...     if isinstance(val, (tuple, list, dict)):
        ...         pass
        ...     else:
        ...         val = val.replace('XXXX', 'CLIENT1')
        ...         section[newkey] = val
        >>> cfg.walk(transform, call_on_sections=True)
        {'CLIENT1section': {'CLIENT1key': None}}
        >>> cfg
        ConfigObj({'CLIENT1section': {'CLIENT1key': 'CLIENT1value'}})
        """
        out = {}
        # scalars first
        for i in range(len(self.scalars)):
            entry = self.scalars[i]
            try:
                val = function(self, entry, **keywargs)
                # bound again in case name has changed
                entry = self.scalars[i]
                out[entry] = val
            except Exception:
                if raise_errors:
                    raise
                else:
                    entry = self.scalars[i]
                    out[entry] = False
        # then sections
        for i in range(len(self.sections)):
            entry = self.sections[i]
            if call_on_sections:
                try:
                    function(self, entry, **keywargs)
                except Exception:
                    if raise_errors:
                        raise
                    else:
                        entry = self.sections[i]
                        out[entry] = False
                # bound again in case name has changed
                entry = self.sections[i]
            # previous result is discarded
            out[entry] = self[entry].walk(
                function,
                raise_errors=raise_errors,
                call_on_sections=call_on_sections,
                **keywargs)
        return out
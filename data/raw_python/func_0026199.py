def validate(self, validator, preserve_errors=False, copy=False,
                 section=None):
        """
        Test the ConfigObj against a configspec.

        It uses the ``validator`` object from *validate.py*.

        To run ``validate`` on the current ConfigObj, call: ::

            test = config.validate(validator)

        (Normally having previously passed in the configspec when the ConfigObj
        was created - you can dynamically assign a dictionary of checks to the
        ``configspec`` attribute of a section though).

        It returns ``True`` if everything passes, or a dictionary of
        pass/fails (True/False). If every member of a subsection passes, it
        will just have the value ``True``. (It also returns ``False`` if all
        members fail).

        In addition, it converts the values from strings to their native
        types if their checks pass (and ``stringify`` is set).

        If ``preserve_errors`` is ``True`` (``False`` is default) then instead
        of a marking a fail with a ``False``, it will preserve the actual
        exception object. This can contain info about the reason for failure.
        For example the ``VdtValueTooSmallError`` indicates that the value
        supplied was too small. If a value (or section) is missing it will
        still be marked as ``False``.

        You must have the validate module to use ``preserve_errors=True``.

        You can then use the ``flatten_errors`` function to turn your nested
        results dictionary into a flattened list of failures - useful for
        displaying meaningful error messages.
        """
        if section is None:
            if self.configspec is None:
                raise ValueError('No configspec supplied.')
            if preserve_errors:
                # We do this once to remove a top level dependency on the validate module
                # Which makes importing configobj faster
                from .validate import VdtMissingValue
                self._vdtMissingValue = VdtMissingValue

            section = self

            if copy:
                section.initial_comment = section.configspec.initial_comment
                section.final_comment = section.configspec.final_comment
                section.encoding = section.configspec.encoding
                section.BOM = section.configspec.BOM
                section.newlines = section.configspec.newlines
                section.indent_type = section.configspec.indent_type

        #
        # section.default_values.clear() #??
        configspec = section.configspec
        self._set_configspec(section, copy)


        def validate_entry(entry, spec, val, missing, ret_true, ret_false):
            section.default_values.pop(entry, None)

            try:
                section.default_values[entry] = validator.get_default_value(configspec[entry])
            except (KeyError, AttributeError, validator.baseErrorClass):
                # No default, bad default or validator has no 'get_default_value'
                # (e.g. SimpleVal)
                pass

            try:
                check = validator.check(spec,
                                        val,
                                        missing=missing
                                        )
            except validator.baseErrorClass as e:
                if not preserve_errors or isinstance(e, self._vdtMissingValue):
                    out[entry] = False
                else:
                    # preserve the error
                    out[entry] = e
                    ret_false = False
                ret_true = False
            else:
                ret_false = False
                out[entry] = True
                if self.stringify or missing:
                    # if we are doing type conversion
                    # or the value is a supplied default
                    if not self.stringify:
                        if isinstance(check, (list, tuple)):
                            # preserve lists
                            check = [self._str(item) for item in check]
                        elif missing and check is None:
                            # convert the None from a default to a ''
                            check = ''
                        else:
                            check = self._str(check)
                    if (check != val) or missing:
                        section[entry] = check
                if not copy and missing and entry not in section.defaults:
                    section.defaults.append(entry)
            return ret_true, ret_false

        #
        out = {}
        ret_true = True
        ret_false = True

        unvalidated = [k for k in section.scalars if k not in configspec]
        incorrect_sections = [k for k in configspec.sections if k in section.scalars]
        incorrect_scalars = [k for k in configspec.scalars if k in section.sections]

        for entry in configspec.scalars:
            if entry in ('__many__', '___many___'):
                # reserved names
                continue
            if (not entry in section.scalars) or (entry in section.defaults):
                # missing entries
                # or entries from defaults
                missing = True
                val = None
                if copy and entry not in section.scalars:
                    # copy comments
                    section.comments[entry] = (
                        configspec.comments.get(entry, []))
                    section.inline_comments[entry] = (
                        configspec.inline_comments.get(entry, ''))
                #
            else:
                missing = False
                val = section[entry]

            ret_true, ret_false = validate_entry(entry, configspec[entry], val,
                                                 missing, ret_true, ret_false)

        many = None
        if '__many__' in configspec.scalars:
            many = configspec['__many__']
        elif '___many___' in configspec.scalars:
            many = configspec['___many___']

        if many is not None:
            for entry in unvalidated:
                val = section[entry]
                ret_true, ret_false = validate_entry(entry, many, val, False,
                                                     ret_true, ret_false)
            unvalidated = []

        for entry in incorrect_scalars:
            ret_true = False
            if not preserve_errors:
                out[entry] = False
            else:
                ret_false = False
                msg = 'Value %r was provided as a section' % entry
                out[entry] = validator.baseErrorClass(msg)
        for entry in incorrect_sections:
            ret_true = False
            if not preserve_errors:
                out[entry] = False
            else:
                ret_false = False
                msg = 'Section %r was provided as a single value' % entry
                out[entry] = validator.baseErrorClass(msg)

        # Missing sections will have been created as empty ones when the
        # configspec was read.
        for entry in section.sections:
            # FIXME: this means DEFAULT is not copied in copy mode
            if section is self and entry == 'DEFAULT':
                continue
            if section[entry].configspec is None:
                unvalidated.append(entry)
                continue
            if copy:
                section.comments[entry] = configspec.comments.get(entry, [])
                section.inline_comments[entry] = configspec.inline_comments.get(entry, '')
            check = self.validate(validator, preserve_errors=preserve_errors, copy=copy, section=section[entry])
            out[entry] = check
            if check == False:
                ret_true = False
            elif check == True:
                ret_false = False
            else:
                ret_true = False

        section.extra_values = unvalidated
        if preserve_errors and not section._created:
            # If the section wasn't created (i.e. it wasn't missing)
            # then we can't return False, we need to preserve errors
            ret_false = False
        #
        if ret_false and preserve_errors and out:
            # If we are preserving errors, but all
            # the failures are from missing sections / values
            # then we can return False. Otherwise there is a
            # real failure that we need to preserve.
            ret_false = not any(out.values())
        if ret_true:
            return True
        elif ret_false:
            return False
        return out
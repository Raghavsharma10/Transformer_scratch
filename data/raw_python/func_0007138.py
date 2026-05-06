def _trim_dictionary_parameters(self, dict_param):
        """Return a dict that only has matching entries in the msgid."""
        # NOTE(luisg): Here we trim down the dictionary passed as parameters
        # to avoid carrying a lot of unnecessary weight around in the message
        # object, for example if someone passes in Message() % locals() but
        # only some params are used, and additionally we prevent errors for
        # non-deepcopyable objects by unicoding() them.

        # Look for %(param) keys in msgid;
        # Skip %% and deal with the case where % is first character on the line
        keys = re.findall('(?:[^%]|^)?%\((\w*)\)[a-z]', self.msgid)

        # If we don't find any %(param) keys but have a %s
        if not keys and re.findall('(?:[^%]|^)%[a-z]', self.msgid):
            # Apparently the full dictionary is the parameter
            params = self._copy_param(dict_param)
        else:
            params = {}
            # Save our existing parameters as defaults to protect
            # ourselves from losing values if we are called through an
            # (erroneous) chain that builds a valid Message with
            # arguments, and then does something like "msg % kwds"
            # where kwds is an empty dictionary.
            src = {}
            if isinstance(self.params, dict):
                src.update(self.params)
            src.update(dict_param)
            for key in keys:
                params[key] = self._copy_param(src[key])

        return params
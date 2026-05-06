def variable_state(cls, scripts, variables):
        """Return the initialization state for each variable in variables.

        The state is determined based on the scripts passed in via the scripts
        parameter.

        If there is more than one 'when green flag clicked' script and they
        both modify the attribute, then the attribute is considered to not be
        initialized.

        """
        def conditionally_set_not_modified():
            """Set the variable to modified if it hasn't been altered."""
            state = variables.get(block.args[0], None)
            if state == cls.STATE_NOT_MODIFIED:
                variables[block.args[0]] = cls.STATE_MODIFIED

        green_flag, other = partition_scripts(scripts, cls.HAT_GREEN_FLAG)
        variables = dict((x, cls.STATE_NOT_MODIFIED) for x in variables)
        for script in green_flag:
            in_zone = True
            for name, level, block in cls.iter_blocks(script.blocks):
                if name == 'broadcast %s and wait':
                    in_zone = False
                if name == 'set %s effect to %s':
                    state = variables.get(block.args[0], None)
                    if state is None:
                        continue  # Not a variable we care about
                    if in_zone and level == 0:  # Success!
                        if state == cls.STATE_NOT_MODIFIED:
                            state = cls.STATE_INITIALIZED
                        else:  # Multiple when green flag clicked conflict
                            # TODO: Need to allow multiple sets of a variable
                            # within the same script
                            # print 'CONFLICT', script
                            state = cls.STATE_MODIFIED
                    elif in_zone:
                        continue  # Conservative ignore for nested absolutes
                    elif state == cls.STATE_NOT_MODIFIED:
                        state = cls.STATE_MODIFIED
                    variables[block.args[0]] = state
                elif name == 'change %s effect by %s':
                    conditionally_set_not_modified()
        for script in other:
            for name, _, block in cls.iter_blocks(script.blocks):
                if name in ('change %s effect by %s', 'set %s effect to %s'):
                    conditionally_set_not_modified()
        return variables
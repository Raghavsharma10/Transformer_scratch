def attribute_state(cls, scripts, attribute):
        """Return the state of the scripts for the given attribute.

        If there is more than one 'when green flag clicked' script and they
        both modify the attribute, then the attribute is considered to not be
        initialized.

        """
        green_flag, other = partition_scripts(scripts, cls.HAT_GREEN_FLAG, cls.HAT_CLONE)
        block_set = cls.BLOCKMAPPING[attribute]
        state = cls.STATE_NOT_MODIFIED
        # TODO: Any regular broadcast blocks encountered in the initialization
        # zone should be added to this loop for conflict checking.
        for script in green_flag:
            in_zone = True
            for name, level, _ in cls.iter_blocks(script.blocks):
                if name == 'broadcast %s and wait':
                    # TODO: Follow the broadcast and wait scripts that occur in
                    # the initialization zone
                    in_zone = False
                if (name, 'absolute') in block_set:
                    if in_zone and level == 0:  # Success!
                        if state == cls.STATE_NOT_MODIFIED:
                            state = cls.STATE_INITIALIZED
                        else:  # Multiple when green flag clicked conflict
                            state = cls.STATE_MODIFIED
                    elif in_zone:
                        continue  # Conservative ignore for nested absolutes
                    else:
                        state = cls.STATE_MODIFIED
                    break  # The state of the script has been determined
                elif (name, 'relative') in block_set:
                    state = cls.STATE_MODIFIED
                    break
        if state != cls.STATE_NOT_MODIFIED:
            return state
        # Check the other scripts to see if the attribute was ever modified
        for script in other:
            for name, _, _ in cls.iter_blocks(script.blocks):
                if name in [x[0] for x in block_set]:
                    return cls.STATE_MODIFIED
        return cls.STATE_NOT_MODIFIED
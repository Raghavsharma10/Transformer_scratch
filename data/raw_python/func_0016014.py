def kill_chain_phase_names(instance):
    """Ensure the `kill_chain_name` and `phase_name` properties of
    `kill_chain_phase` objects follow naming style conventions.
    """
    if instance['type'] in enums.KILL_CHAIN_PHASE_USES and 'kill_chain_phases' in instance:
        for phase in instance['kill_chain_phases']:

            if 'kill_chain_name' not in phase:
                # Since this field is required, schemas will already catch the error
                return

            chain_name = phase['kill_chain_name']
            if not chain_name.islower() or '_' in chain_name or ' ' in chain_name:
                yield JSONError("kill_chain_name '%s' should be all lowercase"
                                " and use hyphens instead of spaces or "
                                "underscores as word separators." % chain_name,
                                instance['id'], 'kill-chain-names')

            phase_name = phase['phase_name']
            if not phase_name.islower() or '_' in phase_name or ' ' in phase_name:
                yield JSONError("phase_name '%s' should be all lowercase and "
                                "use hyphens instead of spaces or underscores "
                                "as word separators." % phase_name,
                                instance['id'], 'kill-chain-names')
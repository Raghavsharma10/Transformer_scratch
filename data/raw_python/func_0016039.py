def list_shoulds(options):
    """Construct the list of 'SHOULD' validators to be run by the validator.
    """
    validator_list = []

    # Default: enable all
    if not options.disabled and not options.enabled:
        validator_list.extend(CHECKS['all'])
        return validator_list

    # --disable
    # Add SHOULD requirements to the list unless disabled
    if options.disabled:
        if 'all' not in options.disabled:
            if 'format-checks' not in options.disabled:
                if 'custom-prefix' not in options.disabled:
                    validator_list.append(CHECKS['custom-prefix'])
                elif 'custom-prefix-lax' not in options.disabled:
                    validator_list.append(CHECKS['custom-prefix-lax'])
                if 'open-vocab-format' not in options.disabled:
                    validator_list.append(CHECKS['open-vocab-format'])
                if 'kill-chain-names' not in options.disabled:
                    validator_list.append(CHECKS['kill-chain-names'])
                if 'observable-object-keys' not in options.disabled:
                    validator_list.append(CHECKS['observable-object-keys'])
                if 'observable-dictionary-keys' not in options.disabled:
                    validator_list.append(CHECKS['observable-dictionary-keys'])
                if 'windows-process-priority-format' not in options.disabled:
                    validator_list.append(CHECKS['windows-process-priority-format'])
                if 'hash-length' not in options.disabled:
                    validator_list.append(CHECKS['hash-length'])

            if 'approved-values' not in options.disabled:
                if 'marking-definition-type' not in options.disabled:
                    validator_list.append(CHECKS['marking-definition-type'])
                if 'relationship-types' not in options.disabled:
                    validator_list.append(CHECKS['relationship-types'])
                if 'duplicate-ids' not in options.disabled:
                    validator_list.append(CHECKS['duplicate-ids'])
                if 'all-vocabs' not in options.disabled:
                    if 'attack-motivation' not in options.disabled:
                        validator_list.append(CHECKS['attack-motivation'])
                    if 'attack-resource-level' not in options.disabled:
                        validator_list.append(CHECKS['attack-resource-level'])
                    if 'identity-class' not in options.disabled:
                        validator_list.append(CHECKS['identity-class'])
                    if 'indicator-types' not in options.disabled:
                        validator_list.append(CHECKS['indicator-types'])
                    if 'industry-sector' not in options.disabled:
                        validator_list.append(CHECKS['industry-sector'])
                    if 'malware-types' not in options.disabled:
                        validator_list.append(CHECKS['malware-types'])
                    if 'report-types' not in options.disabled:
                        validator_list.append(CHECKS['report-types'])
                    if 'threat-actor-types' not in options.disabled:
                        validator_list.append(CHECKS['threat-actor-types'])
                    if 'threat-actor-role' not in options.disabled:
                        validator_list.append(CHECKS['threat-actor-role'])
                    if 'threat-actor-sophistication' not in options.disabled:
                        validator_list.append(CHECKS['threat-actor-sophistication'])
                    if 'tool-types' not in options.disabled:
                        validator_list.append(CHECKS['tool-types'])
                    if 'region' not in options.disabled:
                        validator_list.append(CHECKS['region'])
                    if 'hash-algo' not in options.disabled:
                        validator_list.append(CHECKS['hash-algo'])
                    if 'windows-pebinary-type' not in options.disabled:
                        validator_list.append(CHECKS['windows-pebinary-type'])
                    if 'account-type' not in options.disabled:
                        validator_list.append(CHECKS['account-type'])
                if 'all-external-sources' not in options.disabled:
                    if 'mime-type' not in options.disabled:
                        validator_list.append(CHECKS['mime-type'])
                    if 'protocols' not in options.disabled:
                        validator_list.append(CHECKS['protocols'])
                    if 'ipfix' not in options.disabled:
                        validator_list.append(CHECKS['ipfix'])
                    if 'http-request-headers' not in options.disabled:
                        validator_list.append(CHECKS['http-request-headers'])
                    if 'socket-options' not in options.disabled:
                        validator_list.append(CHECKS['socket-options'])
                    if 'pdf-doc-info' not in options.disabled:
                        validator_list.append(CHECKS['pdf-doc-info'])
                    if 'countries' not in options.disabled:
                        validator_list.append(CHECKS['countries'])

            if 'network-traffic-ports' not in options.disabled:
                validator_list.append(CHECKS['network-traffic-ports'])
            if 'extref-hashes' not in options.disabled:
                validator_list.append(CHECKS['extref-hashes'])

    # --enable
    if options.enabled:
        for check in options.enabled:
            try:
                if CHECKS[check] in validator_list:
                    continue

                if type(CHECKS[check]) is list:
                    validator_list.extend(CHECKS[check])
                else:
                    validator_list.append(CHECKS[check])
            except KeyError:
                raise JSONError("%s is not a valid check!" % check)

    return validator_list
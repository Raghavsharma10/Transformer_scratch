def find_bridges(prior_bridges=None):
    """ Confirm or locate IP addresses of Philips Hue bridges.

    `prior_bridges` -- optional list of bridge serial numbers
    * omitted - all discovered bridges returned as dictionary
    * single string - returns IP as string or None
    * dictionary - validate provided ip's before attempting discovery
    * collection or sequence - return dictionary of filtered sn:ip pairs
      * if mutable then found bridges are removed from argument
    """
    found_bridges = {}

    # Validate caller's provided list
    try:
        prior_bridges_list = prior_bridges.items()
    except AttributeError:
        # if caller didnt provide dict then assume single SN or None
        # in either case, the discovery must be executed
        run_discovery = True
    else:
        for prior_sn, prior_ip in prior_bridges_list:
            if prior_ip:
                serial, baseip = parse_description_xml(_build_from(prior_ip))
                if serial:
                    # there is a bridge at provided IP, add to found
                    found_bridges[serial] = baseip
                else:
                    # nothing usable at that ip
                    logger.info('%s not found at %s', prior_sn, prior_ip)
        run_discovery = found_bridges.keys() != prior_bridges.keys()

    # prior_bridges is None, unknown, dict of unfound SNs, or empty dict
    # found_bridges is dict of found SNs from prior, or empty dict
    if run_discovery:
        # do the discovery, not all IPs were confirmed
        try:
            found_bridges.update(via_upnp())
        except DiscoveryError:
            try:
                found_bridges.update(via_nupnp())
            except DiscoveryError:
                try:
                    found_bridges.update(via_scan())
                except DiscoveryError:
                    logger.warning("All discovery methods returned nothing")

    if prior_bridges:
        # prior_bridges is either single SN or dict of unfound SNs
        # first assume single Serial SN string
        try:
            ip_address = found_bridges[prior_bridges]
        except TypeError:
            # user passed an invalid type for key
            # presumably it's a dict meant for alternate mode
            logger.debug('Assuming alternate mode, prior_bridges is type %s.',
                          type(prior_bridges))
        except KeyError:
            # user provided Serial Number was not found
            # TODO: dropping tuples here if return none executed
            # return None
            pass # let it turn the string into a set, eww
        else:
            # user provided Serial Number found
            return ip_address

        # Filter the found list to subset of prior
        prior_bridges_keys = set(prior_bridges)
        keys_to_remove = prior_bridges_keys ^ found_bridges.keys()
        logger.debug('Removing %s from found_bridges', keys_to_remove)
        for key in keys_to_remove:
            found_bridges.pop(key, None)

        # Filter the prior dict to unfound only
        keys_to_remove = prior_bridges_keys & found_bridges.keys()
        logger.debug('Removing %s from prior_bridges', keys_to_remove)
        for key in keys_to_remove:
            try:
                prior_bridges.pop(key, None)
            except TypeError:
                # not a dict, try as set or list
                prior_bridges.remove(key)
            except AttributeError:
                # likely not mutable
                break

        keys_to_report = prior_bridges_keys - found_bridges.keys()
        for serial in keys_to_report:
            logger.warning('Could not locate bridge with Serial ID %s', serial)

    else:
        # prior_bridges is None or empty dict, return all found
        pass

    return found_bridges
def remote_command(session, command, vehicle_index, poll=True):
    """Send a remote command."""
    if command not in SUPPORTED_COMMANDS:
        raise MoparError("unsupported command: " + command)
    profile = get_profile(session)
    _validate_vehicle(vehicle_index, profile)
    if command in [COMMAND_LOCK, COMMAND_UNLOCK]:
        url = REMOTE_LOCK_COMMAND_URL
    elif command in [COMMAND_ENGINE_ON, COMMAND_ENGINE_OFF]:
        url = REMOTE_ENGINE_COMMAND_URL
    elif command == COMMAND_HORN:
        url = REMOTE_ALARM_COMMAND_URL
    resp = session.post(url, {
        'pin': session.auth.pin,
        'uuid': profile['vehicles'][vehicle_index]['uuid'],
        'action': command
    }).json()
    if poll:
        uuid = profile['vehicles'][vehicle_index]['uuid']
        service_id = resp['serviceRequestId']
        return _remote_status(session, service_id, uuid, url)
    return 'submitted'
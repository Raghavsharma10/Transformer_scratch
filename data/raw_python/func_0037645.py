def filter_commands(commands, invalid_query_starts=('DROP', 'UNLOCK', 'LOCK')):
    """
    Remove particular queries from a list of SQL commands.

    :param commands: List of SQL commands
    :param invalid_query_starts: Type of SQL command to remove
    :return: Filtered list of SQL commands
    """
    commands_with_drops = len(commands)
    filtered_commands = [c for c in commands if not c.startswith(invalid_query_starts)]
    if commands_with_drops - len(filtered_commands) > 0:
        print("\t" + str(invalid_query_starts) + " commands removed", commands_with_drops - len(filtered_commands))
    return filtered_commands
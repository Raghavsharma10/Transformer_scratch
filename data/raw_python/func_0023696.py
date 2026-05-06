def execute_status(args, root_dir=None):
    """Print the status of the daemon.

    This function displays the current status of the daemon as well
    as the whole queue and all available information about every entry
    in the queue.
    `terminaltables` is used to format and display the queue contents.
    `colorclass` is used to color format the various items in the queue.

    Args:
        root_dir (string): The path to the root directory the daemon is running in.

    """
    status = command_factory('status')({}, root_dir=root_dir)
    # First rows, showing daemon status
    if status['status'] == 'running':
        status['status'] = Color('{autogreen}' + '{}'.format(status['status']) + '{/autogreen}')
    elif status['status'] in ['paused']:
        status['status'] = Color('{autoyellow}' + '{}'.format(status['status']) + '{/autoyellow}')

    print('Daemon: {}\n'.format(status['status']))

    # Handle queue data
    data = status['data']
    if isinstance(data, str):
        print(data)
    elif isinstance(data, dict):
        # Format incomming data to be compatible with Terminaltables
        formatted_data = []
        formatted_data.append(['Index', 'Status', 'Code',
                               'Command', 'Path', 'Start', 'End'])
        for key, entry in sorted(data.items(), key=operator.itemgetter(0)):
            formatted_data.append(
                [
                    '#{}'.format(key),
                    entry['status'],
                    '{}'.format(entry['returncode']),
                    entry['command'],
                    entry['path'],
                    entry['start'],
                    entry['end']
                ]
            )

        # Create AsciiTable instance and define style
        table = AsciiTable(formatted_data)
        table.outer_border = False
        table.inner_column_border = False

        terminal_width = terminal_size()
        customWidth = table.column_widths
        # If the text is wider than the actual terminal size, we
        # compute a new size for the Command and Path column.
        if (reduce(lambda a, b: a+b, table.column_widths) + 10) > terminal_width[0]:
            # We have to subtract 14 because of table paddings
            left_space = math.floor((terminal_width[0] - customWidth[0] - customWidth[1] - customWidth[2] - customWidth[5] - customWidth[6] - 14)/2)

            if customWidth[3] < left_space:
                customWidth[4] = 2*left_space - customWidth[3]
            elif customWidth[4] < left_space:
                customWidth[3] = 2*left_space - customWidth[4]
            else:
                customWidth[3] = left_space
                customWidth[4] = left_space

        # Format long strings to match the console width
        for i, entry in enumerate(table.table_data):
            for j, string in enumerate(entry):
                max_width = customWidth[j]
                wrapped_string = '\n'.join(wrap(string, max_width))
                if j == 1:
                    if wrapped_string == 'done' or wrapped_string == 'running' or wrapped_string == 'paused':
                        wrapped_string = Color('{autogreen}' + '{}'.format(wrapped_string) + '{/autogreen}')
                    elif wrapped_string in ['queued', 'stashed']:
                        wrapped_string = Color('{autoyellow}' + '{}'.format(wrapped_string) + '{/autoyellow}')
                    elif wrapped_string in ['failed', 'stopping', 'killing']:
                        wrapped_string = Color('{autored}' + '{}'.format(wrapped_string) + '{/autored}')
                elif j == 2:
                    if wrapped_string == '0' and wrapped_string != 'Code':
                        wrapped_string = Color('{autogreen}' + '{}'.format(wrapped_string) + '{/autogreen}')
                    elif wrapped_string != '0' and wrapped_string != 'Code':
                        wrapped_string = Color('{autored}' + '{}'.format(wrapped_string) + '{/autored}')

                table.table_data[i][j] = wrapped_string

        print(table.table)
    print('')
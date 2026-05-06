def _parse_transaction_entry(entry):
    """ Validate & parse a transaction into (date, action, value) tuple. """
    parts = entry.split()

    date_string = parts[0]
    try:
        date = datetime.datetime.strptime(date_string[:-1], '%Y-%m-%d').date()
    except ValueError:
        raise ValueError('Invalid date in vacationrc for entry: {}'.format(entry))

    if len(parts) < 2:
        raise ValueError('.vacationrc missing an action for entry: {}'.format(entry))
    action = parts[1].lower()
    if action not in ('days', 'rate', 'off', 'adjust', 'show'):
        raise ValueError('Invalid action in vacationrc for entry: {}'.format(entry))

    try:
        value = float(parts[2])
    except IndexError:
        value = None
    except (ValueError, TypeError):
        raise ValueError('Invalid value in vacationrc for entry: {}'.format(entry))

    return (date, action, value)
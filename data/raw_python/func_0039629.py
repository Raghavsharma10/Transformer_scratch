def get_days_off(transactions):
    """ Return the dates for any 'take day off' transactions. """
    days_off = []
    for trans in transactions:
        date, action, _ = _parse_transaction_entry(trans)
        if action == 'off':
            days_off.append(date)
    return days_off
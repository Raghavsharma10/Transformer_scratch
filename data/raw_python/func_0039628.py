def sum_transactions(transactions):
    """ Sums transactions into a total of remaining vacation days. """
    workdays_per_year = 250
    previous_date = None
    rate = 0
    day_sum = 0
    for transaction in transactions:
        date, action, value = _parse_transaction_entry(transaction)
        if previous_date is None:
            previous_date = date
        elapsed = workdays.networkdays(previous_date, date, stat_holidays()) - 1

        if action == 'rate':
            rate = float(value) / workdays_per_year
        elif action == 'off':
            elapsed -= 1  # Didn't work that day
            day_sum -= 1  # And we used a day
        day_sum += rate * elapsed

        if action == 'days':
            day_sum = value  # Fixed value as of this entry

        previous_date = date

    return day_sum
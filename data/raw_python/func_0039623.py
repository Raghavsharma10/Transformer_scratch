def sort(transactions):
    """ Return a list of sorted transactions by date. """
    return transactions.sort(key=lambda x: datetime.datetime.strptime(x.split(':')[0], '%Y-%m-%d'))[:]
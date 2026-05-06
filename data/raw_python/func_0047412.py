def total(usaf, field='GHI (W/m^2)'):
    """total annual insolation, defaults to GHI."""
    running_total = 0
    usafdata = data(usaf)
    for record in usafdata:
        running_total += float(record[field])
    return running_total/1000.
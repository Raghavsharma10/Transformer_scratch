def time_choices():
    """Return digital time choices every half hour from 00:00 to 23:30."""
    hours = list(range(0, 24))
    times = []
    for h in hours:
        hour = str(h).zfill(2)
        times.append(hour+':00')
        times.append(hour+':30')
    return list(zip(times, times))
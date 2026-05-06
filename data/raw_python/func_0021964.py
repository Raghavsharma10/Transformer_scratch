def generate_table_from(data):
    "Output a nicely formatted ascii table"
    table = Texttable(max_width=120)
    table.add_row(["view", "method", "status", "count", "minimum", "maximum", "mean", "stdev", "queries", "querytime"]) 
    table.set_cols_align(["l", "l", "l", "r", "r", "r", "r", "r", "r", "r"])

    for item in sorted(data):
        mean = round(sum(data[item]['times'])/data[item]['count'], 3)

        mean_sql = round(sum(data[item]['sql'])/data[item]['count'], 3)
        mean_sqltime = round(sum(data[item]['sqltime'])/data[item]['count'], 3)
        
        sdsq = sum([(i - mean) ** 2 for i in data[item]['times']])
        try:
            stdev = '%.2f' % ((sdsq / (len(data[item]['times']) - 1)) ** .5)
        except ZeroDivisionError:
            stdev = '0.00'

        minimum = "%.2f" % min(data[item]['times'])
        maximum = "%.2f" % max(data[item]['times'])
        table.add_row([data[item]['view'], data[item]['method'], data[item]['status'], data[item]['count'], minimum, maximum, '%.3f' % mean, stdev, mean_sql, mean_sqltime])

    return table.draw()
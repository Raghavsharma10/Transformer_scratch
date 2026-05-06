def output_results(results, metric, options):
    """
    Output the results to stdout.

    TODO: add AMPQ support for efficiency
    """
    formatter = options['Formatter']
    context = metric.copy()  # XXX might need to sanitize this
    try:
        context['dimension'] = list(metric['Dimensions'].values())[0]
    except AttributeError:
        context['dimension'] = ''
    for result in results:
        stat_keys = metric['Statistics']
        if not isinstance(stat_keys, list):
            stat_keys = [stat_keys]
        for statistic in stat_keys:
            context['statistic'] = statistic
            # get and then sanitize metric name, first copy the unit name from the
            # result to the context to keep the default format happy
            context['Unit'] = result['Unit']
            metric_name = (formatter % context).replace('/', '.').lower()
            line = '{0} {1} {2}\n'.format(
                metric_name,
                result[statistic],
                timegm(result['Timestamp'].timetuple()),
            )
            sys.stdout.write(line)
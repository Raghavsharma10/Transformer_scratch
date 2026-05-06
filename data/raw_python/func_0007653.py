def query(scope, blueprint, debug, output, with_metadata, realtime, **description):
    """
    e.g.

        googleanalytics --identity debrouwere --account debrouwere --webproperty http://debrouwere.org \
            query pageviews \
            --start yesterday --limit -10 --sort -pageviews \
            --dimensions pagepath \
            --debug

    """

    if realtime:
        description['type'] = 'realtime'

    if blueprint:
        queries = from_blueprint(scope, blueprint)
    else:
        if not isinstance(scope, ga.account.Profile):
            raise ValueError("Account and webproperty needed for query.")

        queries = from_args(scope, **description)

    for query in queries:
        if debug:
            click.echo(query.build())

        report = query.serialize(format=output, with_metadata=with_metadata)
        click.echo(report)
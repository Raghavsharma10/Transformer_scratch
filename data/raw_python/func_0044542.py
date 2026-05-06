def index():
    """ Base testsuite view. """

    # setup_env()

    logs = Table('log', metadata, autoload=True)

    criticals = logs.select().where(logs.c.log_level == 50).order_by(
        'siteconfig', 'date_created')
    criticals_count = logs.count(logs.c.log_level == 50)

    errors = logs.select().where(logs.c.log_level == 40).order_by(
        'siteconfig', 'date_created')
    errors_count = logs.count(logs.c.log_level == 40)

    warnings = logs.select().where(logs.c.log_level == 30).order_by(
        'siteconfig', 'date_created')
    warnings_count = logs.count(logs.c.log_level == 30)

    infos = logs.select().where(logs.c.log_level == 20).order_by(
        'siteconfig', 'date_created')
    infos_count = logs.count(logs.c.log_level == 20)

    return render_template(
        'index.html',
        log_criticals=criticals.execute(),
        log_criticals_count=criticals_count.execute().first()[0],
        log_errors=errors.execute(),
        log_errors_count=errors_count.execute().first()[0],
        log_warnings=warnings.execute(),
        log_warnings_count=warnings_count.execute().first()[0],
        log_infos=infos.execute(),
        log_infos_count=infos_count.execute().first()[0],
    )
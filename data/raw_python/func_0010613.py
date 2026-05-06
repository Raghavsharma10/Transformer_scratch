def workflow_status(obj, details):
    """ Show the status of the workflows. """
    show_colors = obj['show_color']
    config_cli = obj['config'].cli

    if details:
        temp_form = '{:>{}}  {:20} {:25} {:25} {:38} {}'
    else:
        temp_form = '{:>{}}  {:20} {:25} {} {} {}'

    click.echo('\n')
    click.echo(temp_form.format(
        'Status',
        12,
        'Name',
        'Start Time',
        'ID' if details else '',
        'Job' if details else '',
        'Arguments'
    ))
    click.echo('-' * (138 if details else 75))

    def print_jobs(jobs, *, label='', color='green'):
        for job in jobs:
            start_time = job.start_time if job.start_time is not None else 'unknown'

            click.echo(temp_form.format(
                _style(show_colors, label, fg=color, bold=True),
                25 if show_colors else 12,
                job.name,
                start_time.replace(tzinfo=pytz.utc).astimezone().strftime(
                    config_cli['time_format']),
                job.workflow_id if details else '',
                job.id if details else '',
                ','.join(['{}={}'.format(k, v) for k, v in job.arguments.items()]))
            )

    # running jobs
    print_jobs(list_jobs(config=obj['config'],
                         status=JobStatus.Active,
                         filter_by_type=JobType.Workflow),
               label='Running', color='green')

    # scheduled jobs
    print_jobs(list_jobs(config=obj['config'],
                         status=JobStatus.Scheduled,
                         filter_by_type=JobType.Workflow),
               label='Scheduled', color='blue')

    # registered jobs
    print_jobs(list_jobs(config=obj['config'],
                         status=JobStatus.Registered,
                         filter_by_type=JobType.Workflow),
               label='Registered', color='yellow')

    # reserved jobs
    print_jobs(list_jobs(config=obj['config'],
                         status=JobStatus.Reserved,
                         filter_by_type=JobType.Workflow),
               label='Reserved', color='yellow')
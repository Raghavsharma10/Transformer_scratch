def worker_status(obj, filter_queues, details):
    """ Show the status of all running workers. """
    show_colors = obj['show_color']

    f_queues = filter_queues.split(',') if filter_queues is not None else None

    workers = list_workers(config=obj['config'], filter_by_queues=f_queues)
    if len(workers) == 0:
        click.echo('No workers are running at the moment.')
        return

    for ws in workers:
        click.echo('{} {}'.format(_style(show_colors, 'Worker:', fg='blue', bold=True),
                                  _style(show_colors, ws.name, fg='blue')))
        click.echo('{:23} {}'.format(_style(show_colors, '> pid:', bold=True), ws.pid))

        if details:
            click.echo('{:23} {}'.format(_style(show_colors, '> concurrency:', bold=True),
                                         ws.concurrency))
            click.echo('{:23} {}'.format(_style(show_colors, '> processes:', bold=True),
                                         ', '.join(str(p) for p in ws.process_pids)))
            click.echo('{:23} {}://{}:{}/{}'.format(_style(show_colors, '> broker:',
                                                           bold=True),
                                                    ws.broker.transport,
                                                    ws.broker.hostname,
                                                    ws.broker.port,
                                                    ws.broker.virtual_host))

        click.echo('{:23} {}'.format(_style(show_colors, '> queues:', bold=True),
                                     ', '.join([q.name for q in ws.queues])))

        if details:
            click.echo('{:23} {}'.format(_style(show_colors, '> job count:', bold=True),
                                         ws.job_count))

            jobs = list_jobs(config=obj['config'], filter_by_worker=ws.name)
            click.echo('{:23} [{}]'.format(_style(show_colors, '> jobs:', bold=True),
                                           len(jobs) if len(jobs) > 0 else 'No tasks'))

            for job in jobs:
                click.echo('{:15} {} {}'.format(
                    '',
                    _style(show_colors, '{}'.format(job.name),
                           bold=True, fg=JOB_COLOR[job.type]),
                    _style(show_colors, '({}) [{}] <{}> on {}'.format(
                        job.type, job.workflow_id, job.id, job.worker_pid),
                        fg=JOB_COLOR[job.type])))

        click.echo('\n')
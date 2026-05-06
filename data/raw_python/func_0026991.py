def log_celery_task(request):
    """ Add description to celery log output """
    task = request.task
    description = None
    if isinstance(task, Task):
        try:
            description = task.get_description(*request.args, **request.kwargs)
        except NotImplementedError:
            pass
        except Exception as e:
            # Logging should never break workflow.
            logger.exception('Cannot get description for task %s. Error: %s' % (task.__class__.__name__, e))

    return '{0.name}[{0.id}]{1}{2}{3}'.format(
        request,
        ' {0}'.format(description) if description else '',
        ' eta:[{0}]'.format(request.eta) if request.eta else '',
        ' expires:[{0}]'.format(request.expires) if request.expires else '',
    )
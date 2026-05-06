def create_app(config):
    """ Create a fully configured Celery application object.

    Args:
        config (Config): A reference to a lightflow configuration object.

    Returns:
        Celery: A fully configured Celery application object.
    """

    # configure the celery logging system with the lightflow settings
    setup_logging.connect(partial(_initialize_logging, config), weak=False)
    task_postrun.connect(partial(_cleanup_workflow, config), weak=False)

    # patch Celery to use cloudpickle instead of pickle for serialisation
    patch_celery()

    # create the main celery app and load the configuration
    app = Celery('lightflow')
    app.conf.update(**config.celery)

    # overwrite user supplied settings to make sure celery works with lightflow
    app.conf.update(
        task_serializer='pickle',
        accept_content=['pickle'],
        result_serializer='pickle',
        task_default_queue=DefaultJobQueueName.Task
    )

    if isinstance(app.conf.include, list):
        app.conf.include.extend(LIGHTFLOW_INCLUDE)
    else:
        if len(app.conf.include) > 0:
            raise ConfigOverwriteError(
                'The content in the include config will be overwritten')
        app.conf.include = LIGHTFLOW_INCLUDE

    return app
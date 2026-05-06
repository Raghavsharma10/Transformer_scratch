def send_task(app_label, task_name):
    """ A helper function to deal with waldur_core "high-level" tasks.
        Define high-level task with explicit name using a pattern:
        waldur_core.<app_label>.<task_name>

        .. code-block:: python
            @shared_task(name='waldur_core.openstack.provision_instance')
            def provision_instance_fn(instance_uuid, backend_flavor_id)
                pass

        Call it by name:

        .. code-block:: python
            send_task('openstack', 'provision_instance')(instance_uuid, backend_flavor_id)

        Which is identical to:

        .. code-block:: python
            provision_instance_fn.delay(instance_uuid, backend_flavor_id)

    """

    def delay(*args, **kwargs):
        full_task_name = 'waldur_core.%s.%s' % (app_label, task_name)
        send_celery_task(full_task_name, args, kwargs, countdown=2)

    return delay
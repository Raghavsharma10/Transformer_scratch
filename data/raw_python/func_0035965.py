def order_stop(backend, order_run_id):
    """
    Stop the run of an order - Stop the running order and keep all state around.
    """
    if order_run_id is None:
        raise click.ClickException('invalid order id %s' % order_run_id)

    click.secho('%s - Stop order id %s' % (get_datetime(), order_run_id), fg='green')
    check_and_print(DKCloudCommandRunner.stop_orderrun(backend.dki, order_run_id.strip()))
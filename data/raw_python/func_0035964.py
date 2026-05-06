def order_stop(backend, order_id):
    """
    Stop an order - Turn off the serving generation ability of an order.  Stop any running jobs.  Keep all state around.
    """
    if order_id is None:
        raise click.ClickException('invalid order id %s' % order_id)
    click.secho('%s - Stop order id %s' % (get_datetime(), order_id), fg='green')
    check_and_print(DKCloudCommandRunner.stop_order(backend.dki, order_id))
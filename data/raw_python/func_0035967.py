def delete_orderrun(backend, orderrun_id):
    """
    Delete the orderrun specified by the argument.
    """
    click.secho('%s - Deleting orderrun %s' % (get_datetime(), orderrun_id), fg='green')
    check_and_print(DKCloudCommandRunner.delete_orderrun(backend.dki, orderrun_id.strip()))
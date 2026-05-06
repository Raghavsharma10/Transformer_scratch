def destroy_rackspace(connection, region, instance_id):
    """ terminates the instance """

    server = connection.servers.get(instance_id)
    log_yellow('deleting rackspace instance ...')
    server.delete()

    # wait for server to be deleted
    try:
        while True:
            server = connection.servers.get(server.id)
            log_yellow('waiting for deletion ...')
            sleep(5)
    except:
        pass
    log_green('The server has been deleted')
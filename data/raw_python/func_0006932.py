def destroy_rackspace(region, instance_id, access_key_id, secret_access_key):
    """ terminates the instance """
    nova = connect_to_rackspace(region,
                                access_key_id,
                                secret_access_key)

    server = nova.servers.get(instance_id)
    log_yellow('deleting rackspace instance ...')
    server.delete()

    # wait for server to be deleted
    try:
        while True:
            server = nova.servers.get(server.id)
            log_yellow('waiting for deletion ...')
            sleep(5)
    except:
        pass
    log_green('The server has been deleted')
    os.unlink('data.json')
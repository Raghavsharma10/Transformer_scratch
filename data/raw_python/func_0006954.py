def print_rackspace_info(region,
                         instance_id,
                         access_key_id,
                         secret_access_key,
                         username):
    """ outputs information about our Rackspace instance """
    data = get_rackspace_info(server_id=instance_id,
                              region=region,
                              access_key_id=access_key_id,
                              secret_access_key=secret_access_key,
                              username=username)

    log_green("region: %s" % data['region'])
    log_green("Instance state: %s" % data['state'])
    log_green("Ip address: %s" % data['ip_address'])
    log_green("volume: %s" % data['volume'])
    log_green("user: %s" % data['username'])
    log_green("ssh -i %s %s@%s" % (env.key_filename,
                                   username,
                                   data['ip_address']))
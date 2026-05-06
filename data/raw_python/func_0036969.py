def add_bare_metal_cloud(client, cloud, keys):
    """
    Black magic is happening here. All of this wil change when we sanitize our API, however, this works until then
    """
    title = cloud.get('title')
    provider = cloud.get('provider')
    key = cloud.get('apikey', "")
    secret = cloud.get('apisecret', "")
    tenant_name = cloud.get('tenant_name', "")
    region = cloud.get('region', "")
    apiurl = cloud.get('apiurl', "")
    compute_endpoint = cloud.get('compute_endpoint', None)
    machine_ip = cloud.get('machine_ip', None)
    machine_key = cloud.get('machine_key', None)
    machine_user = cloud.get('machine_user', None)
    machine_port = cloud.get('machine_port', None)

    if provider == "bare_metal":
        machine_ids = cloud['machines'].keys()
        bare_machine = cloud['machines'][machine_ids[0]]
        machine_hostname = bare_machine.get('dns_name', None)
        if not machine_hostname:
            machine_hostname = bare_machine['public_ips'][0]

        if not machine_ip:
            machine_ip = machine_hostname
        key = machine_hostname
        machine_name = cloud['machines'][machine_ids[0]]['name']
        machine_id = machine_ids[0]

        keypairs = keys.keys()
        for i in keypairs:
            keypair_machines = keys[i]['machines']
            if keypair_machines:
                keypair_machs = keys[i]['machines']
                for mach in keypair_machs:
                    if mach[1] == machine_id:
                        machine_key = i
                        break
            else:
                pass

    client.add_cloud(title, provider, key, secret, tenant_name=tenant_name, region=region, apiurl=apiurl,
                       machine_ip=machine_ip, machine_key=machine_key, machine_user=machine_user,
                       compute_endpoint=compute_endpoint, machine_port=machine_port)
def prepare_monitor(tenant=tenant, user=user, password=password, organization=organization, zone_name=zone_name):
    """
    :param tenant: tenant url
    :param user: user's email
    :param password: user's password
    :param zone_name: (optional) zone_name
    :return:
    """
    router = PrivatePath(tenant, verify_codes=False)

    payload = {
        "firstName": "AllSeeingEye",
        "lastName": "Monitor",
        "email": user,
        "password": password,
        "accept": "true"
    }
    try:
        router.post_quick_sign_up(data=payload)
    except exceptions.ApiUnauthorizedError:
        pass

    platform = QubellPlatform.connect(tenant=tenant, user=user, password=password)
    org = platform.organization(name=organization)
    if zone_name:
        zone = org.zones[zone_name]
    else:
        zone = org.zone
    env = org.environment(name="Monitor for "+zone.name, zone=zone.id)
    env.init_common_services(with_cloud_account=False, zone_name=zone_name)

    # todo: move to env
    policy_name = lambda policy: "{}.{}".format(policy.get('action'), policy.get('parameter'))
    env_data = env.json()
    key_id = [p for p in env_data['policies'] if 'provisionVms.publicKeyId' == policy_name(p)][0].get('value')

    with env as envbulk:
        envbulk.add_marker('monitor')
        envbulk.add_property('publicKeyId', 'string', key_id)

    monitor = Manifest(file=os.path.join(os.path.dirname(__file__), './monitor_manifests/monitor.yml'))
    monitor_child = Manifest(file=os.path.join(os.path.dirname(__file__), './monitor_manifests/monitor_child.yml'))

    org.application(manifest=monitor_child, name='monitor-child')
    app = org.application(manifest=monitor, name='monitor')

    return platform, org.id, app.id, env.id
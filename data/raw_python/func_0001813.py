def harvest(lancet, config_section):
    """Construct a new Harvest client."""
    url, username, password = lancet.get_credentials(
        config_section, credentials_checker
    )

    project_id_getter = lancet.get_instance_from_config(
        "timer", "project_id_getter", lancet
    )
    task_id_getter = lancet.get_instance_from_config(
        "timer", "task_id_getter", lancet
    )

    client = HarvestPlatform(
        server=url,
        basic_auth=(username, password),
        project_id_getter=project_id_getter,
        task_id_getter=task_id_getter,
    )
    lancet.call_on_close(client.close)
    return client
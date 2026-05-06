def provision_system_config(items, database_name, overwrite=False, clear=False, skip_user_check=False):
    """Provision a basic system configuration"""

    from hfos.provisions.base import provisionList
    from hfos.database import objectmodels

    default_system_config_count = objectmodels['systemconfig'].count({
        'name': 'Default System Configuration'})

    if default_system_config_count == 0 or (clear or overwrite):
        provisionList([SystemConfiguration], 'systemconfig', overwrite, clear, skip_user_check)
        hfoslog('Provisioning: System: Done.', emitter='PROVISIONS')
    else:
        hfoslog('Default system configuration already present.', lvl=warn,
                emitter='PROVISIONS')
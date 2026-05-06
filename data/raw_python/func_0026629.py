def provision_system_vessel(items, database_name, overwrite=False, clear=False, skip_user_check=False):
    """Provisions the default system vessel"""

    from hfos.provisions.base import provisionList
    from hfos.database import objectmodels

    vessel = objectmodels['vessel'].find_one({'name': 'Default System Vessel'})
    if vessel is not None:
        if overwrite is False:
            hfoslog('Default vessel already existing. Skipping provisions.')
            return
        else:
            vessel.delete()

    provisionList([SystemVessel], 'vessel', overwrite, clear, skip_user_check)

    sysconfig = objectmodels['systemconfig'].find_one({'active': True})
    hfoslog('Adapting system config for default vessel:', sysconfig)
    sysconfig.vesseluuid = SystemVessel['uuid']
    sysconfig.save()

    hfoslog('Provisioning: Vessel: Done.', emitter='PROVISIONS')
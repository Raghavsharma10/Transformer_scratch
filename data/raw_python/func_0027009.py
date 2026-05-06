def _is_active_model(cls, model):
        """ Check is model app name is in list of INSTALLED_APPS """
        # We need to use such tricky way to check because of inconsistent apps names:
        # some apps are included in format "<module_name>.<app_name>" like "waldur_core.openstack"
        # other apps are included in format "<app_name>" like "nodecondcutor_sugarcrm"
        return ('.'.join(model.__module__.split('.')[:2]) in settings.INSTALLED_APPS or
                '.'.join(model.__module__.split('.')[:1]) in settings.INSTALLED_APPS)
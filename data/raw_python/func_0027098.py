def _get_erred_shared_settings_module(self):
        """
        Returns a LinkList based module which contains link to shared service setting instances in ERRED state.
        """
        result_module = modules.LinkList(title=_('Shared provider settings in erred state'))
        result_module.template = 'admin/dashboard/erred_link_list.html'
        erred_state = structure_models.SharedServiceSettings.States.ERRED

        queryset = structure_models.SharedServiceSettings.objects
        settings_in_erred_state = queryset.filter(state=erred_state).count()

        if settings_in_erred_state:
            result_module.title = '%s (%s)' % (result_module.title, settings_in_erred_state)
            for service_settings in queryset.filter(state=erred_state).iterator():
                module_child = self._get_link_to_instance(service_settings)
                module_child['error'] = service_settings.error_message
                result_module.children.append(module_child)
        else:
            result_module.pre_content = _('Nothing found.')

        return result_module
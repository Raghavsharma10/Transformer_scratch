def _get_erred_resources_module(self):
        """
        Returns a list of links to resources which are in ERRED state and linked to a shared service settings.
        """
        result_module = modules.LinkList(title=_('Resources in erred state'))
        erred_state = structure_models.NewResource.States.ERRED
        children = []

        resource_models = SupportedServices.get_resource_models()
        resources_in_erred_state_overall = 0
        for resource_type, resource_model in resource_models.items():
            queryset = resource_model.objects.filter(service_project_link__service__settings__shared=True)
            erred_amount = queryset.filter(state=erred_state).count()
            if erred_amount:
                resources_in_erred_state_overall = resources_in_erred_state_overall + erred_amount
                link = self._get_erred_resource_link(resource_model, erred_amount, erred_state)
                children.append(link)

        if resources_in_erred_state_overall:
            result_module.title = '%s (%s)' % (result_module.title, resources_in_erred_state_overall)
            result_module.children = children
        else:
            result_module.pre_content = _('Nothing found.')

        return result_module
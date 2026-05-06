def get_related_models(cls, model):
        """ Get a dictionary with related structure models for given class or model:

            >> SupportedServices.get_related_models(gitlab_models.Project)
            {
                'service': nodeconductor_gitlab.models.GitLabService,
                'service_project_link': nodeconductor_gitlab.models.GitLabServiceProjectLink,
                'resources': [
                    nodeconductor_gitlab.models.Group,
                    nodeconductor_gitlab.models.Project,
                ]
            }
        """
        from waldur_core.structure.models import ServiceSettings

        if isinstance(model, ServiceSettings):
            model_str = cls._registry.get(model.type, {}).get('model_name', '')
        else:
            model_str = cls._get_model_str(model)

        for models in cls.get_service_models().values():
            if model_str == cls._get_model_str(models['service']) or \
               model_str == cls._get_model_str(models['service_project_link']):
                return models

            for resource_model in models['resources']:
                if model_str == cls._get_model_str(resource_model):
                    return models
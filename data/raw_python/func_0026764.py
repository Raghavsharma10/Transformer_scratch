def ziggurat_model_init(
    user=None,
    group=None,
    user_group=None,
    group_permission=None,
    user_permission=None,
    user_resource_permission=None,
    group_resource_permission=None,
    resource=None,
    external_identity=None,
    *args,
    **kwargs
):
    """
    This function handles attaching model to service if model has one specified
    as `_ziggurat_service`, Also attached a proxy object holding all model
    definitions that services might use

    :param args:
    :param kwargs:
    :param passwordmanager, the password manager to override default one
    :param passwordmanager_schemes, list of schemes for default
            passwordmanager to use
    :return:
    """
    models = ModelProxy()
    models.User = user
    models.Group = group
    models.UserGroup = user_group
    models.GroupPermission = group_permission
    models.UserPermission = user_permission
    models.UserResourcePermission = user_resource_permission
    models.GroupResourcePermission = group_resource_permission
    models.Resource = resource
    models.ExternalIdentity = external_identity

    model_service_mapping = import_model_service_mappings()

    if kwargs.get("passwordmanager"):
        user.passwordmanager = kwargs["passwordmanager"]
    else:
        user.passwordmanager = make_passwordmanager(
            kwargs.get("passwordmanager_schemes")
        )

    for name, cls in models.items():
        # if model has a manager attached attached the class also to manager
        services = model_service_mapping.get(name, [])
        for service in services:
            setattr(service, "model", cls)
            setattr(service, "models_proxy", models)
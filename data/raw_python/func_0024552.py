def get_model_objects(model, wfi_role=None, **kwargs):
        """
        Fetches model objects by filtering with kwargs

        If wfi_role is specified, then we expect kwargs contains a
        filter value starting with role,

        e.g. {'user': 'role.program.user'}

        We replace this `role` key with role instance parameter `wfi_role` and try to get
        object that filter value 'role.program.user' points by iterating `getattribute`. At
        the end filter argument becomes {'user': user}.

        Args:
            model (Model): Model class
            wfi_role (Role): role instance of wf instance
            **kwargs: filter arguments

        Returns:
            (list): list of model object instances
        """
        query_dict = {}
        for k, v in kwargs.items():
            if isinstance(v, list):
                query_dict[k] = [str(x) for x in v]
            else:
                parse = str(v).split('.')
                if parse[0] == 'role' and wfi_role:
                    query_dict[k] = wfi_role
                    for i in range(1, len(parse)):
                        query_dict[k] = query_dict[k].__getattribute__(parse[i])
                else:
                    query_dict[k] = parse[0]

        return model.objects.all(**query_dict)
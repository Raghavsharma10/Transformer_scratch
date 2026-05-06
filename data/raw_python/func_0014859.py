def connect(self):
        """Iterate through the application configuration and instantiate
        the services.
        """
        requested_services = set(
            svc.lower() for svc in current_app.config.get('BOTO3_SERVICES', [])
        )

        region = current_app.config.get('BOTO3_REGION')
        sess_params = {
            'aws_access_key_id': current_app.config.get('BOTO3_ACCESS_KEY'),
            'aws_secret_access_key': current_app.config.get('BOTO3_SECRET_KEY'),
            'profile_name': current_app.config.get('BOTO3_PROFILE'),
            'region_name': region
        }
        sess = boto3.session.Session(**sess_params)

        try:
            cns = {}
            for svc in requested_services:
                # Check for optional parameters
                params = current_app.config.get(
                    'BOTO3_OPTIONAL_PARAMS', {}
                ).get(svc, {})

                # Get session params and override them with kwargs
                # `profile_name` cannot be passed to clients and resources
                kwargs = sess_params.copy()
                kwargs.update(params.get('kwargs', {}))
                del kwargs['profile_name']

                # Override the region if one is defined as an argument
                args = params.get('args', [])
                if len(args) >= 1:
                    del kwargs['region_name']

                if not(isinstance(args, list) or isinstance(args, tuple)):
                    args = [args]

                # Create resource or client
                if svc in sess.get_available_resources():
                    cns.update({svc: sess.resource(svc, *args, **kwargs)})
                else:
                    cns.update({svc: sess.client(svc, *args, **kwargs)})
        except UnknownServiceError:
            raise
        return cns
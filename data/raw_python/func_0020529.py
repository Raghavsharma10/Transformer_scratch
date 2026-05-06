def create_worker_build(self, **kwargs):
        """
        Create a worker build

        Pass through method to create_prod_build with the following
        modifications:
            - platform param is required
            - release param is required
            - arrangement_version param is required, which is used to
              select which worker_inner:n.json template to use
            - inner template set to worker_inner:n.json if not set
            - outer template set to worker.json if not set
            - customize configuration set to worker_customize.json if not set

        :return: BuildResponse instance
        """
        missing = set()
        for required in ('platform', 'release', 'arrangement_version'):
            if not kwargs.get(required):
                missing.add(required)

        if missing:
            raise ValueError("Worker build missing required parameters: %s" %
                             missing)

        if kwargs.get('platforms'):
            raise ValueError("Worker build called with unwanted platforms param")

        arrangement_version = kwargs['arrangement_version']
        kwargs.setdefault('inner_template', WORKER_INNER_TEMPLATE.format(
            arrangement_version=arrangement_version))
        kwargs.setdefault('outer_template', WORKER_OUTER_TEMPLATE)
        kwargs.setdefault('customize_conf', WORKER_CUSTOMIZE_CONF)
        kwargs['build_type'] = BUILD_TYPE_WORKER
        try:
            return self._do_create_prod_build(**kwargs)
        except IOError as ex:
            if os.path.basename(ex.filename) == kwargs['inner_template']:
                raise OsbsValidationException("worker invalid arrangement_version %s" %
                                              arrangement_version)

            raise
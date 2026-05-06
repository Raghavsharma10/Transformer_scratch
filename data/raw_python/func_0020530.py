def create_orchestrator_build(self, **kwargs):
        """
        Create an orchestrator build

        Pass through method to create_prod_build with the following
        modifications:
            - platforms param is required
            - arrangement_version param may be used to select which
              orchestrator_inner:n.json template to use
            - inner template set to orchestrator_inner:n.json if not set
            - outer template set to orchestrator.json if not set
            - customize configuration set to orchestrator_customize.json if not set

        :return: BuildResponse instance
        """
        if not self.can_orchestrate():
            raise OsbsOrchestratorNotEnabled("can't create orchestrate build "
                                             "when can_orchestrate isn't enabled")
        extra = [x for x in ('platform',) if kwargs.get(x)]
        if extra:
            raise ValueError("Orchestrator build called with unwanted parameters: %s" %
                             extra)

        arrangement_version = kwargs.setdefault('arrangement_version',
                                                self.build_conf.get_arrangement_version())

        if arrangement_version < REACTOR_CONFIG_ARRANGEMENT_VERSION and not kwargs.get('platforms'):
            raise ValueError('Orchestrator build requires platforms param')

        kwargs.setdefault('inner_template', ORCHESTRATOR_INNER_TEMPLATE.format(
            arrangement_version=arrangement_version))
        kwargs.setdefault('outer_template', ORCHESTRATOR_OUTER_TEMPLATE)
        kwargs.setdefault('customize_conf', ORCHESTRATOR_CUSTOMIZE_CONF)
        kwargs['build_type'] = BUILD_TYPE_ORCHESTRATOR
        try:
            return self._do_create_prod_build(**kwargs)
        except IOError as ex:
            if os.path.basename(ex.filename) == kwargs['inner_template']:
                raise OsbsValidationException("orchestrator invalid arrangement_version %s" %
                                              arrangement_version)

            raise
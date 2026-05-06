def _deferred_blueprint_init(self, setup_state):
        """Bind resources to the app as recorded in blueprint.

        Synchronize prefix between blueprint/api and registration options, then
        perform initialization with setup_state.app :class:`flask.Flask` object.
        When a :class:`flask.ext.resteasy.Api` object is initialized with a blueprint,
        this method is recorded on the blueprint to be run when the blueprint is later
        registered to a :class:`flask.Flask` object.  This method also monkeypatches
        BlueprintSetupState.add_url_rule with _add_url_rule_patch.

        :param setup_state: The setup state object passed to deferred functions
            during blueprint registration
        :type setup_state: :class:`flask.blueprints.BlueprintSetupState`
        """
        self.blueprint_setup = setup_state
        if setup_state.add_url_rule.__name__ != '_add_url_rule_patch':
            setup_state._original_add_url_rule = setup_state.add_url_rule
            setup_state.add_url_rule = MethodType(Api._add_url_rule_patch,
                                                  setup_state)
        if not setup_state.first_registration:
            raise ValueError('flask-RESTEasy blueprints can only be registered once.')
        self._init_app(setup_state.app)
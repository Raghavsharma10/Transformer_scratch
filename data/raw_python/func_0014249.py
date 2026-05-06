def ready(self):
        '''Called by Django when the app is ready for use.'''
        # set up the options
        self.options = {}
        self.options.update(DEFAULT_OPTIONS)
        for template_engine in settings.TEMPLATES:
            if template_engine.get('BACKEND', '').startswith('django_mako_plus'):
                self.options.update(template_engine.get('OPTIONS', {}))

        # dmp-enabled apps registry
        self.registration_lock = threading.RLock()
        self.registered_apps = {}

        # init the template engine
        self.engine = engines['django_mako_plus']

        # default imports on every compiled template
        self.template_imports = [
            'import django_mako_plus',
            'import django.utils.html',     # used in template.py
        ]
        self.template_imports.extend(self.options['DEFAULT_TEMPLATE_IMPORTS'])

        # initialize the list of providers
        ProviderRun.initialize_providers()

        # set up the parameter converters (can't import until apps are set up)
        from .converter.base import ParameterConverter
        ParameterConverter._sort_converters(app_ready=True)
def _add_new_init_method(cls):
    """
    Replace the existing cls.__init__() method with a new one
    which calls the original one and in addition performs the
    following actions:

    (1) Finds all instances of tohu.BaseGenerator in the namespace
        and collects them in the dictionary `self.field_gens`.
    (2) ..to do..
    """

    orig_init = cls.__init__

    def new_init_method(self, *args, **kwargs):
        logger.debug(f"Initialising new {self} (type: {type(self)})")

        # Call original __init__ function to ensure we pick up
        # any tohu generators that are defined there.
        #
        logger.debug(f"    orig_init: {orig_init}")
        orig_init(self, *args, **kwargs)

        #
        # Find field generator templates and spawn them to create
        # field generators for the new custom generator instance.
        #
        field_gens_templates = find_field_generator_templates(self)
        logger.debug(f'Found {len(field_gens_templates)} field generator template(s):')
        debug_print_dict(field_gens_templates)

        logger.debug('Spawning field generator templates...')
        origs = {}
        spawned = {}
        dependency_mapping = {}
        for (name, gen) in field_gens_templates.items():
            origs[name] = gen
            spawned[name] = gen.spawn(dependency_mapping)
            logger.debug(f'Adding dependency mapping: {gen} -> {spawned[name]}')

        self.field_gens = spawned
        self.__dict__.update(self.field_gens)

        logger.debug(f'Spawned field generators attached to custom generator instance:')
        debug_print_dict(self.field_gens)

        # Add seed generator
        #
        #self.seed_generator = SeedGenerator()

        # Create class for the items produced by this generator
        #
        self.__class__.item_cls = make_item_class_for_custom_generator_class(self)

    cls.__init__ = new_init_method
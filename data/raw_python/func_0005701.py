def add_new_init_method(obj):
    """
    Replace the existing obj.__init__() method with a new one
    which calls the original one and in addition performs the
    following actions:

    (1) Finds all instances of tohu.BaseGenerator in the namespace
        and collects them in the dictionary `self.field_gens`.
    (2) ..to do..
    """

    orig_init = obj.__init__

    def new_init(self, *args, **kwargs):
        logger.debug(f"Initialising new {self}")

        # Call original __init__ function to ensure we pick up
        # any tohu generators that are defined there.
        orig_init(self, *args, **kwargs)

        #
        # Find field generator templates and attach spawned copies
        #
        field_gens_templates = find_field_generators(self)
        logger.debug(f'Found {len(field_gens_templates)} field generator template(s):')
        debug_print_dict(field_gens_templates)

        def find_orig_parent(dep_gen, origs):
            """
            Find name and instance of the parent of the dependent
            generator `dep_gen` amongst the generators in `origs`.
            """
            for parent_name, parent in origs.items():
                if dep_gen.parent is parent:
                    return parent_name, parent
            raise RuntimeError(f"Parent of dependent generator {dep_gen} not defined in the same custom generator")


        logger.debug('Spawning field generator templates...')
        origs = {}
        spawned = {}
        for name, gen in field_gens_templates.items():
            if isinstance(gen, IndependentGenerator) and gen in origs.values():
                logger.debug(f'Cloning generator {name}={gen} because it is an alias for an existing generator')
                gen = gen.clone()

            if isinstance(gen, IndependentGenerator):
                origs[name] = gen
                spawned[name] = gen._spawn()
                logger.debug(f'Spawning generator {gen}. New spawn: {spawned[name]}')
            elif isinstance(gen, DependentGenerator):
                orig_parent_name, orig_parent = find_orig_parent(gen, origs)
                new_parent = spawned[orig_parent_name]
                #spawned[name] = new_parent.clone()
                spawned[name] = gen._spawn_and_reattach_parent(new_parent)
            else:
                pass

        self.field_gens = spawned
        self.__dict__.update(self.field_gens)

        logger.debug(f'Field generators attached to custom generator instance:')
        debug_print_dict(self.field_gens)

        #
        # Add seed generator
        #
        self.seed_generator = SeedGenerator()

        #
        # Create class for the items produced by this generator
        #
        self.__class__.item_cls = make_item_class_for_custom_generator(self)

    obj.__init__ = new_init
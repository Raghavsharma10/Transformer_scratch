def augment_init_method(cls):
    """
    Replace the existing cls.__init__() method with a new one which
    also initialises the field generators and similar bookkeeping.
    """

    orig_init = cls.__init__

    def new_init(self, *args, **kwargs):
        super(CustomGenerator, self).__init__()  # TODO: does this behave correctly with longer inheritance chains?

        orig_init(self, *args, **kwargs)

        self.orig_args = args
        self.orig_kwargs = kwargs

        self.ns_gen_templates = TohuNamespace()
        self.ns_gen_templates.update_from_dict(self.__class__.__dict__)
        self.ns_gen_templates.update_from_dict(self.__dict__)
        self.ns_gen_templates.set_owner(self.__class__)
        self._mark_field_generator_templates()

        self.ns_gens = self.ns_gen_templates.spawn()
        self.ns_gens.set_owner(self)

        self._update_namespace_with_field_generators()
        self._set_field_names()
        self._set_tohu_items_name()
        self._set_tohu_items_cls()

    cls.__init__ = new_init
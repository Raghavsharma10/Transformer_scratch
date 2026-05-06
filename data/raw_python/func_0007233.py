def _make_persistent(self, model_name, pkg_name):
        """Monkey-patch object persistence (ex: to/from database) into a
        bravado-core model class"""

        #
        # WARNING: ugly piece of monkey-patching below. Hopefully will replace
        # with native bravado-core code in the future...
        #

        # Load class at path pkg_name
        c = get_function(pkg_name)
        for name in ('load_from_db', 'save_to_db'):
            if not hasattr(c, name):
                raise PyMacaronCoreException("Class %s has no static method '%s'" % (pkg_name, name))

        log.info("Making %s persistent via %s" % (model_name, pkg_name))

        # Replace model generator with one that adds 'save_to_db' to every instance
        model = getattr(self.model, model_name)
        n = self._wrap_bravado_model_generator(model, c.save_to_db, pkg_name)
        setattr(self.model, model_name, n)

        # Add class method load_from_db to model generator
        model = getattr(self.model, model_name)
        setattr(model, 'load_from_db', c.load_from_db)
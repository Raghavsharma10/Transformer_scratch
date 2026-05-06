def bind(self, model, *, skip_table_setup=False):
        """Create backing tables for a model and its non-abstract subclasses.

        :param model: Base model to bind.  Can be abstract.
        :param skip_table_setup: Don't create or verify the table in DynamoDB.  Default is False.
        :raises bloop.exceptions.InvalidModel: if ``model`` is not a subclass of :class:`~bloop.models.BaseModel`.
        """
        # Make sure we're looking at models
        validate_is_model(model)

        concrete = set(filter(lambda m: not m.Meta.abstract, walk_subclasses(model)))
        if not model.Meta.abstract:
            concrete.add(model)
        logger.debug("binding non-abstract models {}".format(
            sorted(c.__name__ for c in concrete)
        ))

        # create_table doesn't block until ACTIVE or validate.
        # It also doesn't throw when the table already exists, making it safe
        # to call multiple times for the same unbound model.
        if skip_table_setup:
            logger.info("skip_table_setup is True; not trying to create tables or validate models during bind")
        else:
            self.session.clear_cache()

        is_creating = {}

        for model in concrete:
            table_name = self._compute_table_name(model)
            before_create_table.send(self, engine=self, model=model)
            if not skip_table_setup:
                if table_name in is_creating:
                    continue
                creating = self.session.create_table(table_name, model)
                is_creating[table_name] = creating

        for model in concrete:
            if not skip_table_setup:
                table_name = self._compute_table_name(model)
                if is_creating[table_name]:
                    # polls until table is active
                    self.session.describe_table(table_name)
                    if model.Meta.ttl:
                        self.session.enable_ttl(table_name, model)
                    if model.Meta.backups and model.Meta.backups["enabled"]:
                        self.session.enable_backups(table_name, model)
                self.session.validate_table(table_name, model)
                model_validated.send(self, engine=self, model=model)
            model_bound.send(self, engine=self, model=model)

        logger.info("successfully bound {} models to the engine".format(len(concrete)))
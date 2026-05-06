def run(self):
        """Run migrations in 'online' mode.
    
        In this scenario we need to create an Engine
        and associate a connection with the context.
    
        """
        connectable = engine_from_config(
            self._config.get_section(self._config.config_ini_section),
            prefix='sqlalchemy.',
            poolclass=pool.NullPool)

        with connectable.connect() as connection:
            ensureSchemaExists(connectable, self._schemaName)

            context.configure(
                connection=connection,
                target_metadata=self._targetMetadata,
                include_object=self._includeObjectFilter,
                include_schemas=True,
                version_table_schema=self._schemaName
            )

            with context.begin_transaction():
                context.run_migrations()
def _execute_migrations(self, current_version, destination_version):
        """
        passed a version:
            this version don't exists in the database and is younger than the last version -> do migrations up until this version
            this version don't exists in the database and is older than the last version -> do nothing, is a unpredictable behavior
            this version exists in the database and is older than the last version -> do migrations down until this version

        didn't pass a version -> do migrations up until the last available version
        """

        is_migration_up = True
        # check if a version was passed to the program
        if self.config.get("schema_version"):
            # if was passed and this version is present in the database, check if is older than the current version
            destination_version_id = self.sgdb.get_version_id_from_version_number(destination_version)
            if destination_version_id:
                current_version_id = self.sgdb.get_version_id_from_version_number(current_version)
                # if this version is previous to the current version in database, then will be done a migration down to this version
                if current_version_id > destination_version_id:
                    is_migration_up = False
            # if was passed and this version is not present in the database and is older than the current version, raise an exception
            # cause is trying to go down to something that never was done
            elif current_version > destination_version:
                raise Exception("Trying to migrate to a lower version wich is not found on database (%s)" % destination_version)

        # getting only the migration sql files to be executed
        migrations_to_be_executed = self._get_migration_files_to_be_executed(current_version, destination_version, is_migration_up)

        self._execution_log("- Current version is: %s" % current_version, "GREEN", log_level_limit=1)

        if migrations_to_be_executed is None or len(migrations_to_be_executed) == 0:
            self._execution_log("- Destination version is: %s" % current_version, "GREEN", log_level_limit=1)
            self._execution_log("\nNothing to do.\n", "PINK", log_level_limit=1)
            return

        self._execution_log("- Destination version is: %s" % (is_migration_up and migrations_to_be_executed[-1].version or destination_version), "GREEN", log_level_limit=1)

        up_down_label = is_migration_up and "up" or "down"
        if self.config.get("show_sql_only", False):
            self._execution_log("\nWARNING: database migrations are not being executed ('--showsqlonly' activated)", "YELLOW", log_level_limit=1)
        else:
            self._execution_log("\nStarting migration %s!" % up_down_label, log_level_limit=1)

        self._execution_log("*** versions: %s\n" % ([ migration.version for migration in migrations_to_be_executed]), "CYAN", log_level_limit=1)

        sql_statements_executed = []
        for migration in migrations_to_be_executed:
            sql = is_migration_up and migration.sql_up or migration.sql_down

            if not self.config.get("show_sql_only", False):
                self._execution_log("===== executing %s (%s) =====" % (migration.file_name, up_down_label), log_level_limit=1)

                label = None
                if is_migration_up:
                    label = self.config.get("label_version", None)

                try:
                    self.sgdb.change(sql, migration.version, migration.file_name, migration.sql_up, migration.sql_down, is_migration_up, self._execution_log, label)
                except Exception as e:
                    self._execution_log("===== ERROR executing %s (%s) =====" % (migration.abspath, up_down_label), log_level_limit=1)
                    raise e

                # paused mode
                if self.config.get("paused_mode", False):
                    if (sys.version_info > (3, 0)):
                        input("* press <enter> to continue... ")
                    else:
                        raw_input("* press <enter> to continue... ")

            # recording the last statement executed
            sql_statements_executed.append(sql)

        if self.config.get("show_sql", False) or self.config.get("show_sql_only", False):
            self._execution_log("__________ SQL statements executed __________", "YELLOW", log_level_limit=1)
            for sql in sql_statements_executed:
                self._execution_log(sql, "YELLOW", log_level_limit=1)
            self._execution_log("_____________________________________________", "YELLOW", log_level_limit=1)
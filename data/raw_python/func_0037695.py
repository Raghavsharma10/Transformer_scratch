def copy_database(self, source, destination):
        """
        Copy a database's content and structure.

        SMALL Database speed improvements (DB size < 5mb)
        Using optimized is about 178% faster
        Using one_query is about 200% faster

        LARGE Database speed improvements (DB size > 5mb)
        Using optimized is about 900% faster
        Using one_query is about 2600% faster

        :param source: Source database
        :param destination: Destination database
        """
        print('\tCopying database {0} structure and data to database {1}'.format(source, destination))
        with Timer('\nSuccess! Copied database {0} to {1} in '.format(source, destination)):
            # Create destination database if it does not exist
            if destination in self.databases:
                self.truncate_database(destination)
            # Truncate database if it does exist
            else:
                self.create_database(destination)

            # Copy database structure and data
            self.change_db(source)
            tables = self.tables

            # Change database to destination
            self.change_db(destination)
            print('\n')
            _enable_printing = self.enable_printing
            self.enable_printing = False

            # Copy tables structure
            for table in tqdm(tables, total=len(tables), desc='Copying {0} table structures'.format(source)):
                self.execute('CREATE TABLE {0}.{1} LIKE {2}.{1}'.format(destination, wrap(table), source))

            # Copy tables data
            for table in tqdm(tables, total=len(tables), desc='Copying {0} table data'.format(source)):
                self.execute('INSERT INTO {0}.{1} SELECT * FROM {2}.{1}'.format(destination, wrap(table), source))
            self.enable_printing = _enable_printing
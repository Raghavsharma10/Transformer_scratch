def save(self, directory=None):
        """
        Dump the entire contents of the database into the tabledata directory as ascii files
        """
        from subprocess import call

        # If user did not supply a new directory, use the one loaded (default: tabledata)
        if isinstance(directory, type(None)):
            directory = self.directory

        # Create the .sql file is it doesn't exist, i.e. if the Database class called a .db file initially
        if not os.path.isfile(self.sqlpath):
            self.sqlpath = self.dbpath.replace('.db', '.sql')
            os.system('touch {}'.format(self.sqlpath))

        # # Write the data to the .sql file
        # with open(self.sqlpath, 'w') as f:
        #     for line in self.conn.iterdump():
        #         f.write('%s\n' % line)

        # Alternatively...
        # Write the schema
        os.system("echo '.output {}\n.schema' | sqlite3 {}".format(self.sqlpath, self.dbpath))

        # Write the table files to the tabledata directory
        os.system("mkdir -p {}".format(directory))
        tables = self.query("select tbl_name from sqlite_master where type='table'")['tbl_name']
        tablepaths = [self.sqlpath]
        for table in tables:
            print('Generating {}...'.format(table))
            tablepath = '{0}/{1}.sql'.format(directory, table)
            tablepaths.append(tablepath)
            with open(tablepath, 'w') as f:
                for line in self.conn.iterdump():
                    line = line.strip()
                    if line.startswith('INSERT INTO "{}"'.format(table)):
                        if sys.version_info.major == 2:
                            f.write(u'{}\n'.format(line).encode('utf-8'))
                        else:
                            f.write(u'{}\n'.format(line))

        print("Tables saved to directory {}/".format(directory))
        print("""=======================================================================================
You can now run git to commit and push these changes, if needed.
For example, if on the master branch you can do the following:
  git add {0} {1}/*.sql
  git commit -m "COMMIT MESSAGE HERE"
  git push origin master
You can then issue a pull request on GitHub to have these changes reviewed and accepted
======================================================================================="""
              .format(self.sqlpath, directory))
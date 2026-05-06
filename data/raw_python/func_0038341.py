def close(self, silent=False):
        """
        Close the database and ask to save and delete the file

        Parameters
        ----------
        silent: bool
            Close quietly without saving or deleting (Default: False).
        """
        if not silent:
            saveme = get_input("Save database contents to '{}/'? (y, [n]) \n"
                               "To save elsewhere, run db.save() before closing. ".format(self.directory))
            if saveme.lower() == 'y':
                self.save()

            delete = get_input("Do you want to delete {0}? (y,[n]) \n"
                               "Don't worry, a new one will be generated if you run astrodb.Database('{1}') "
                               .format(self.dbpath, self.sqlpath))
            if delete.lower() == 'y':
                print("Deleting {}".format(self.dbpath))
                os.system("rm {}".format(self.dbpath))

        print('Closing connection')
        self.conn.close()
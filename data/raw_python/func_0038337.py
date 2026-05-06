def add_changelog(self, user="", mod_tables="", user_desc=""):
        """
        Add an entry to the changelog table. This should be run when changes or edits are done to the database.

        Parameters
        ----------
        user: str
            Name of the person who made the edits
        mod_tables: str
            Table or tables that were edited
        user_desc: str
            A short message describing the changes
        """
        import datetime
        import socket
        
        # Spit out warning messages if the user does not provide the needed information
        if user == "" or mod_tables == "" or user_desc == "":
            print("You must supply your name, the name(s) of table(s) edited, "
                  "and a description for add_changelog() to work.")
            raise InputError('Did not supply the required input, see help(db.add_changelog) for more information.\n'
                             'Your inputs: \n\t user = {}\n\t mod_tables = {}\n\t user_desc = {}'.format(user, mod_tables, user_desc))
                             
        # Making tables all uppercase for consistency
        mod_tables = mod_tables.upper()
        
        data = list()
        data.append(['date', 'user', 'machine_name', 'modified_tables', 'user_description'])
        datestr = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        machine = socket.gethostname()
        data.append([datestr, user, machine, mod_tables, user_desc])
        self.add_data(data, 'changelog')
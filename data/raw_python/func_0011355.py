def initdb(self):
        '''initdb will check for writability of the data folder, meaning
           that it is bound to the local machine. If the folder isn't bound,
           expfactory runs in demo mode (not saving data)
        '''

        self.database = EXPFACTORY_DATABASE
        bot.info("DATABASE: %s" %self.database)

        # Supported database options
        valid = ('sqlite', 'postgres', 'mysql', 'filesystem')
        if not self.database.startswith(valid):
            bot.warning('%s is not yet a supported type, saving to filesystem.' % self.database)
            self.database = 'filesystem'

        # Add functions specific to database type
        self.init_db() # uses url in self.database

        bot.log("Data base: %s" % self.database)
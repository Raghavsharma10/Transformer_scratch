def open_database(self):
        """
        Opens the sqlite database.

        """
        if not self.con:
            try:
                self.con = psycopg2.connect(host=self.host,
                    database=self.dbname, user=self.user,
                    password=self.password, port=self.port)
            except psycopg2.Error as e:
                print("Error while opening database:")
                print(e.pgerror)
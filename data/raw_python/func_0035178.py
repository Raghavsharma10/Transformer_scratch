def create_database(self):
        """
        Creates an empty database if not exists.
        
        """
        if not self._database_exists():
            con = psycopg2.connect(host=self.host, database="postgres",
                user=self.user, password=self.password, port=self.port)
            con.set_isolation_level(
                psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
            query = "CREATE DATABASE {0};".format(self.dbname)
            c = con.cursor()
            c.execute(query)            
            con.close()


            if self.normalize:
                self.open_database()
                query = "CREATE EXTENSION IF NOT EXISTS \"plperlu\";"
                self.execute_sql(query)
    #            query = """CREATE OR REPLACE FUNCTION normalize(str text)
    #RETURNS text
    #AS $$
    #import unicodedata
    #return ''.join(c for c in unicodedata.normalize('NFKD', str)
    #if unicodedata.category(c) != 'Mn')
    #$$ LANGUAGE plpython3u IMMUTABLE;"""
    #             query = """CREATE OR REPLACE FUNCTION normalize(mystr text)
    #   RETURNS text
    # AS $$
    #     from unidecode import unidecode
    #     return unidecode(mystr.decode("utf-8"))
    # $$ LANGUAGE plpythonu IMMUTABLE;"""
                query = """CREATE OR REPLACE FUNCTION normalize(text)
      RETURNS text
    AS $$
        use Text::Unidecode;
        return unidecode(shift);
    $$ LANGUAGE plperlu IMMUTABLE;"""
                self.execute_sql(query)
                self.commit()
                self.close_database()
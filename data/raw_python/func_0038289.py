def get_sql(self):
        """Retrieve the data type for a data record."""
        test_method = [
            self.is_time,
            self.is_date,
            self.is_datetime,
            self.is_decimal,
            self.is_year,
            self.is_tinyint,
            self.is_smallint,
            self.is_mediumint,
            self.is_int,
            self.is_bigint,
            self.is_tinytext,
            self.is_varchar,
            self.is_mediumtext,
            self.is_longtext,
        ]
        # Loop through test methods until a test returns True
        for method in test_method:
            if method():
                return self.sql
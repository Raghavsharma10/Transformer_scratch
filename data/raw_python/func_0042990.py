def get_obstory_ids(self):
        """
        Retrieve the IDs of all obstorys.

        :return:
            A list of obstory IDs for all obstorys
        """
        self.con.execute('SELECT publicId FROM archive_observatories;')
        return map(lambda row: row['publicId'], self.con.fetchall())
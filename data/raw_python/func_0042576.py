def check_for_maintenance(self):
        '''
        Returns True if the maintenance worker should be run now,
        and False otherwise.
        :return:
        '''
        numrevs = self.conn.get_one("SELECT count(revnum) FROM csetLog")[0]
        if numrevs >= SIGNAL_MAINTENACE_CSETS:
            return True
        return False
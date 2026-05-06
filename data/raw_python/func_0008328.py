def update_tcs_table(self):
        """
        Periodically update a table of info from the TCS.

        Only works at GTC
        """
        g = get_root(self).globals
        if not g.cpars['tcs_on'] or not g.cpars['telins_name'].lower() == 'gtc':
            self.after(60000, self.update_tcs_table)
            return

        try:
            tel_server = tcs.get_telescope_server()
            telpars = tel_server.getTelescopeParams()
            add_gtc_header_table_row(self.tcs_table, telpars)
        except Exception as err:
            g.clog.warn('Could not update table of TCS info')

        # schedule next call for 60s later
        self.after(60000, self.update_tcs_table)
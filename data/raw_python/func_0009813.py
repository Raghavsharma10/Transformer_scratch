def get_pin_codes(self, refresh=False):
        """Get the list of PIN codes

        Codes can also be found with self.get_complex_value('PinCodes')
        """
        if refresh:
            self.refresh()
        val = self.get_value("pincodes")

        # val syntax string: <VERSION=3>next_available_user_code_id\tuser_code_id,active,date_added,date_used,PIN_code,name;\t...
        # See (outdated) http://wiki.micasaverde.com/index.php/Luup_UPnP_Variables_and_Actions#DoorLock1

        # Remove the trailing tab
        # ignore the version and next available at the start
        # and split out each set of code attributes
        raw_code_list = []
        try:
            raw_code_list = val.rstrip().split('\t')[1:]
        except Exception as ex:
            logger.error('Got unsupported string {}: {}'.format(val, ex))

        # Loop to create a list of codes
        codes = []
        for code in raw_code_list:

            try:
                # Strip off trailing semicolon
                # Create a list from csv
                code_addrs = code.split(';')[0].split(',')

                # Get the code ID (slot) and see if it should have values
                slot, active = code_addrs[:2]
                if active != '0':
                    # Since it has additional attributes, get the remaining ones
                    _, _, pin, name = code_addrs[2:]
                    # And add them as a tuple to the list
                    codes.append((slot, name, pin))
            except Exception as ex:
                logger.error('Problem parsing pin code string {}: {}'.format(code, ex))
        
        return codes
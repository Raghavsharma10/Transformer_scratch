def get_last_user(self, refresh=False):
        """Get the last used PIN user id"""
        if refresh:
            self.refresh_complex_value('sl_UserCode')
        val = self.get_complex_value("sl_UserCode")
        # Syntax string: UserID="<pin_slot>" UserName="<pin_code_name>"
        # See http://wiki.micasaverde.com/index.php/Luup_UPnP_Variables_and_Actions#DoorLock1

        try:
            # Get the UserID="" and UserName="" fields separately
            raw_userid, raw_username = val.split(' ')
            # Get the right hand value without quotes of UserID="<here>"
            userid = raw_userid.split('=')[1].split('"')[1]
            # Get the right hand value without quotes of UserName="<here>"
            username = raw_username.split('=')[1].split('"')[1]
        except Exception as ex:
            logger.error('Got unsupported user string {}: {}'.format(val, ex))
            return None

        return ( userid, username )
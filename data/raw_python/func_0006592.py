def locale(self):
        '''
        Do a lookup for the locale code that is set for this layout.

        NOTE: USB HID specifies only 35 different locales. If your layout does not fit, it should be set to Undefined/0

        @return: Tuple (<USB HID locale code>, <name>)
        '''
        name = self.json_data['hid_locale']

        # Set to Undefined/0 if not set
        if name is None:
            name = "Undefined"

        return (int(self.json_data['from_hid_locale'][name]), name)
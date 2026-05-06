def __build_sms_data(self, message):
        """Build a dictionary of SMS message elements"""

        attributes = {}

        attributes_to_translate = {
            'to' : 'To',
            'message' : 'Content',
            'client_id' : 'ClientID',
            'concat' : 'Concat',
            'from_name': 'From',
            'invalid_char_option' : 'InvalidCharOption',
            'truncate' : 'Truncate',
            'wrapper_id' : 'WrapperId'
        }

        for attr in attributes_to_translate:
            val_to_use = None
            if hasattr(message, attr):
                val_to_use = getattr(message, attr)
            if val_to_use is None and hasattr(self, attr):
                val_to_use = getattr(self, attr)
            if val_to_use is not None:
                attributes[attributes_to_translate[attr]] = str(val_to_use)

        return attributes
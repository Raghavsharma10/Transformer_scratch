def get_registration_dt(self, use_cached=True):
        """Get the datetime of when this device was added to Device Cloud"""
        device_json = self.get_device_json(use_cached)
        start_date_iso8601 = device_json.get("devRecordStartDate")
        if start_date_iso8601:
            return iso8601_to_dt(start_date_iso8601)
        else:
            return None
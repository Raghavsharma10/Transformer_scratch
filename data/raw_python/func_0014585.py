def chassis_info(self, chassis):
        """Get information about the specified chassis."""
        if not chassis or not isinstance(chassis, str):
            raise RuntimeError('missing chassis address')
        self._check_session()
        status, data = self._rest.get_request('chassis', chassis)
        return data
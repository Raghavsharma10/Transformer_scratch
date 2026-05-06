def save(self):
        """
        IPAddress can only change its PTR record. Saves the current state, PUT /ip_address/uuid.
        """
        body = {'ip_address': {'ptr_record': self.ptr_record}}
        data = self.cloud_manager.request('PUT', '/ip_address/' + self.address, body)
        self._reset(**data['ip_address'])
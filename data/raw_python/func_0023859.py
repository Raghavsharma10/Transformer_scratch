def modify_ip(self, ip_addr, ptr_record):
        """
        Modify an IP address' ptr-record (Reverse DNS).

        Accepts an IPAddress instance (object) or its address (string).
        """
        body = {
            'ip_address': {
                'ptr_record': ptr_record
            }
        }

        res = self.request('PUT', '/ip_address/' + str(ip_addr), body)
        return IPAddress(cloud_manager=self, **res['ip_address'])
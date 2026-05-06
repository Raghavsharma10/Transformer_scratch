def connect_build(self, nyamuk, keepalive, clean_session, retain = 0, dup = 0, version = 3):
        """Build packet for CONNECT command."""
        will = 0; will_topic = None
        byte = 0

        client_id = utf8encode(nyamuk.client_id)
        username  = utf8encode(nyamuk.username) if nyamuk.username is not None else None
        password  = utf8encode(nyamuk.password) if nyamuk.password is not None else None

        #payload len
        payload_len = 2 + len(client_id)
        if nyamuk.will is not None:
            will = 1
            will_topic = utf8encode(nyamuk.will.topic)

            payload_len = payload_len + 2 + len(will_topic) + 2 + nyamuk.will.payloadlen
        
        if username is not None:
            payload_len = payload_len + 2 + len(username)
            if password != None:
                payload_len = payload_len + 2 + len(password)
        
        self.command = NC.CMD_CONNECT
        self.remaining_length = 12 + payload_len
    
        rc = self.alloc()
        if rc != NC.ERR_SUCCESS:
            return rc
         
        #var header
        self.write_string(getattr(NC, 'PROTOCOL_NAME_{0}'.format(version)))
        self.write_byte(  getattr(NC, 'PROTOCOL_VERSION_{0}'.format(version)))
        
        byte = (clean_session & 0x1) << 1
        
        if will:
            byte = byte | ((nyamuk.will.retain & 0x1) << 5) | ((nyamuk.will.qos & 0x3) << 3) | ((will & 0x1) << 2)
        
        if nyamuk.username is not None:
            byte = byte | 0x1 << 7
            if nyamuk.password is not None:
                byte = byte | 0x1 << 6
        
        self.write_byte(byte)
        self.write_uint16(keepalive)
        #payload
        self.write_string(client_id)
        
        if will:
            self.write_string(will_topic)
            self.write_string(nyamuk.will.payload)

        if username is not None:
            self.write_string(username)
            if password is not None:
                self.write_string(password)
            
        nyamuk.keep_alive = keepalive
        
        return NC.ERR_SUCCESS
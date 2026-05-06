def handle_suback(self):
        """Handle incoming SUBACK packet."""
        self.logger.info("SUBACK received")
        
        ret, mid = self.in_packet.read_uint16()
        
        if ret != NC.ERR_SUCCESS:
            return ret
        
        qos_count = self.in_packet.remaining_length - self.in_packet.pos
        granted_qos = bytearray(qos_count)
        
        if granted_qos is None:
            return NC.ERR_NO_MEM
        
        i = 0
        while self.in_packet.pos < self.in_packet.remaining_length:
            ret, byte = self.in_packet.read_byte()
            
            if ret != NC.ERR_SUCCESS:
                granted_qos = None
                return ret
            
            granted_qos[i] = byte
            
            i += 1
        
        evt = event.EventSuback(mid, list(granted_qos))
        self.push_event(evt)
        
        granted_qos = None
        
        return NC.ERR_SUCCESS
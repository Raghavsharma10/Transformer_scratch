def alloc(self):
        """from _mosquitto_packet_alloc."""
        byte = 0
        remaining_bytes = bytearray(5)
        i = 0
        
        remaining_length = self.remaining_length
        
        self.payload = None
        self.remaining_count = 0
        loop_flag = True
        
        #self.dump()
        while loop_flag:
            byte = remaining_length % 128
            remaining_length = remaining_length / 128
            
            if remaining_length > 0:
                byte = byte | 0x80
                
            remaining_bytes[self.remaining_count] = byte
            self.remaining_count += 1
            
            if not (remaining_length > 0 and self.remaining_count < 5):
                loop_flag = False
        
        if self.remaining_count == 5:
            return NC.ERR_PAYLOAD_SIZE
        
        self.packet_length = self.remaining_length + 1 + self.remaining_count
        self.payload = bytearray(self.packet_length)
        
        self.payload[0] = self.command
        
        i = 0
        while i < self.remaining_count:
            self.payload[i+1] = remaining_bytes[i]
            i += 1
        
        self.pos = 1 + self.remaining_count
        
        return NC.ERR_SUCCESS
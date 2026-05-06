def send_subscribe(self, dup, topics):
        """Send subscribe COMMAND to server."""
        pkt = MqttPkt()
        
        pktlen = 2 + sum([2+len(topic)+1 for (topic, qos) in topics])
        pkt.command = NC.CMD_SUBSCRIBE | (dup << 3) | (1 << 1)
        pkt.remaining_length = pktlen
        
        ret = pkt.alloc()
        if ret != NC.ERR_SUCCESS:
            return ret
        
        #variable header
        mid = self.mid_generate()
        pkt.write_uint16(mid)
        
        #payload
        for (topic, qos) in topics:
            pkt.write_string(topic)
            pkt.write_byte(qos)
        
        return self.packet_queue(pkt)
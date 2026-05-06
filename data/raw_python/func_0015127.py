def send_unsubscribe(self, dup, topics):
        """Send unsubscribe COMMAND to server."""
        pkt = MqttPkt()
        
        pktlen = 2 + sum([2+len(topic) for topic in topics])
        pkt.command = NC.CMD_UNSUBSCRIBE | (dup << 3) | (1 << 1)
        pkt.remaining_length = pktlen
        
        ret = pkt.alloc()
        if ret != NC.ERR_SUCCESS:
            return ret
        
        #variable header
        mid = self.mid_generate()
        pkt.write_uint16(mid)
        
        #payload
        for topic in topics:
            pkt.write_string(topic)
        
        return self.packet_queue(pkt)
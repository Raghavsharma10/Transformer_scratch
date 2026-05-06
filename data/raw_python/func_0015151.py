def packet_queue(self, pkt):
        """Enqueue packet to out_packet queue."""
        
        pkt.pos = 0
        pkt.to_process = pkt.packet_length
        
        self.out_packet.append(pkt)
        return NC.ERR_SUCCESS
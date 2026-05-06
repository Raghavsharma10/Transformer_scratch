def packet_write(self):
        """Write packet to network."""
        bytes_written = 0
        
        if self.sock == NC.INVALID_SOCKET:
            return NC.ERR_NO_CONN, bytes_written
        
        while len(self.out_packet) > 0:
            pkt = self.out_packet[0]
            write_length, status = nyamuk_net.write(self.sock, pkt.payload)
            if write_length > 0:
                pkt.to_process -= write_length
                pkt.pos += write_length
                
                bytes_written += write_length
                
                if pkt.to_process > 0:
                    return NC.ERR_SUCCESS, bytes_written
            else:
                if status == errno.EAGAIN or status == errno.EWOULDBLOCK:
                    return NC.ERR_SUCCESS, bytes_written
                elif status == errno.ECONNRESET:
                    return NC.ERR_CONN_LOST, bytes_written
                else:
                    return NC.ERR_UNKNOWN, bytes_written
            
            """
            if pkt.command & 0xF6 == NC.CMD_PUBLISH and self.on_publish is not None:
                self.in_callback = True
                self.on_publish(pkt.mid)
                self.in_callback = False
            """
            
            #next
            del self.out_packet[0]
            
            #free data (unnecessary)
            
            self.last_msg_out = time.time()
            
        
        return NC.ERR_SUCCESS, bytes_written
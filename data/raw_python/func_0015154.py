def build_publish_pkt(self, mid, topic, payload, qos, retain, dup):
        """Build PUBLISH packet."""
        pkt = MqttPkt()
        payloadlen = len(payload)
        packetlen = 2 + len(topic) + payloadlen
        
        if qos > 0:
            packetlen += 2
        
        pkt.mid = mid
        pkt.command = NC.CMD_PUBLISH | ((dup & 0x1) << 3) | (qos << 1) | retain
        pkt.remaining_length = packetlen
        
        ret = pkt.alloc()
        if ret != NC.ERR_SUCCESS:
            return ret, None
        
        #variable header : Topic String
        pkt.write_string(topic)
        
        if qos > 0:
            pkt.write_uint16(mid)
        
        #payloadlen
        if payloadlen > 0:
            pkt.write_bytes(payload, payloadlen)
        
        return NC.ERR_SUCCESS, pkt
def pubrec(self, mid):
        """Send PUBREC response to server."""
        if self.sock == NC.INVALID_SOCKET:
            return NC.ERR_NO_CONN

        self.logger.info("Send PUBREC (msgid=%s)", mid)
        pkt = MqttPkt()

        pkt.command = NC.CMD_PUBREC
        pkt.remaining_length = 2

        ret = pkt.alloc()
        if ret != NC.ERR_SUCCESS:
            return ret

        #variable header: acknowledged message id
        pkt.write_uint16(mid)

        return self.packet_queue(pkt)
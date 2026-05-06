def loop(self, timeout = 1):
        """Main loop."""
        rlist = [self.sock]
        wlist = []
        if len(self.out_packet) > 0:
            wlist.append(self.sock)

        to_read, to_write, _ = select.select(rlist, wlist, [], timeout)
        
        if len(to_read) > 0:
            ret, _ = self.loop_read()
            if ret != NC.ERR_SUCCESS:
                return ret
        
        if len(to_write) > 0:
            ret, _ = self.loop_write()
            if ret != NC.ERR_SUCCESS:
                return ret
            
        self.loop_misc()
        
        return NC.ERR_SUCCESS
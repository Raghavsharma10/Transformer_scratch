def list_current_filter_set(self,raw=False):
        """User to list a currently selected filter set"""
        
        buf = []

        self.open_umanager()
        self.ser.write(''.join((self.cmd_current_filter_list,self.cr)))
        if self.read_loop(lambda x: x.endswith(self.umanager_prompt),self.timeout,lambda x,y,z: buf.append(y.rstrip()[:-1])):
            if raw:
                rv = buf = buf[0]
            else:
                rv, buf = self.filter_organizer(buf[0])
        else:
            raise Dam1021Error(16,"Failed to list currently selected filter set")
        self.close_umanager()

        log.info(buf)

        return rv
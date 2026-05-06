def download(self,data,um_update=False):
        """Used to download firmware or filter set.
        
        :param data: binary string to push via serial 
        :param um_update: flag whether to update umanager
        """
        
        self.open_umanager()
        self.ser.write(''.join((self.cmd_download,self.cr)))
        if self.read_loop(lambda x: x.endswith(self.xmodem_crc),self.timeout):
            if self.xmodem.send(StringIO.StringIO(data)):
                log.info("Data sent")
            else:
                raise Dam1021Error(4,"Error during file download")
        else:
            raise Dam1021Error(3,"uManager is not ready to accept a data")
      
        if self.read_loop(lambda x: x.lower().find(self.reprogram_ack) != -1,self.timeout):
            skr_sum = hashlib.sha1(data).hexdigest()
            log.info("File downloaded. Data SHA-1 checksum: {}".format(skr_sum))
        else:
            raise Dam1021Error(5,"uManager accepted data and not reprogrammed")

        if um_update:
            self.ser.write(''.join((self.cmd_update,self.cr)))
            if self.read_loop(lambda x: x.lower().find(self.update_confirmation) != -1,self.timeout*self.umanager_waitcoeff):
                self.ser.write(self.update_ack)
            else:
                raise Dam1021Error(13,"Error during update command invocation")

            if self.read_loop(lambda x: x.lower().find(self.update_reset) != -1,self.timeout*self.umanager_waitcoeff):
                log.info("uManager updated")
            else:
                raise Dam1021Error(14,"Update failed")
        else:
            self.close_umanager()            


        return skr_sum
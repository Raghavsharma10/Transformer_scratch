def start_server(self):
        """start the selenium Remote Server."""        
        self.__subp = subprocess.Popen(self.command)        
        #print("\tselenium jar pid[%s] is running." %self.__subp.pid)        
        time.sleep(2)
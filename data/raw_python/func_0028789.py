def Request(self, item, timeout=None):
        """Request DDE client
        timeout in seconds
        Note ... handle the exception within this function.
        """
        if not timeout:
            timeout = self.ddetimeout
        try:
            reply = self.ddec.request(item, int(timeout*1000)) # convert timeout into milliseconds
        except DDEError:
            err_str = str(sys.exc_info()[1])
            error = err_str[err_str.find('err=')+4:err_str.find('err=')+10]
            if error == hex(DMLERR_DATAACKTIMEOUT):
                print("TIMEOUT REACHED. Please use a higher timeout.\n")
            if (sys.version_info > (3, 0)): #this is only evaluated in case of an error
                reply = b'-998' #Timeout error value
            else:
                reply = '-998' #Timeout error value
        return reply
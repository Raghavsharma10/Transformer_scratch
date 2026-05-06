def get_exception_error():
        ''' Get the exception info
        Sample usage:
            try:
                raise Exception("asdfsdfsdf")
            except:
                print common.get_exception_error()
        Return:
            return the exception infomation.
        '''
        error_message = ""
        for i in range(len(inspect.trace())):
            error_line = u"""
        File:      %s - [%s]
        Function:  %s
        Statement: %s
        -""" % (inspect.trace()[i][1], inspect.trace()[i][2], inspect.trace()[i][3], inspect.trace()[i][4])
            
            error_message = "%s%s" % (error_message, error_line)    
        
        error_message = u"""Error!\n%s\n\t%s\n\t%s\n-------------------------------------------------------------------------------------------\n\n""" % (error_message,sys.exc_info()[0], sys.exc_info()[1])
        
        return error_message
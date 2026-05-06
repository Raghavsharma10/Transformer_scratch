def error(self, id, errorCode, errorString):
        '''Error during communication with TWS'''
        if errorCode == 165: # Historical data sevice message
            sys.stderr.write("TWS INFO - %s: %s\n" % (errorCode, errorString))
        elif errorCode >= 501 and errorCode < 600: # Socket read failed
            sys.stderr.write("TWS CLIENT-ERROR - %s: %s\n" % (errorCode, errorString))
        elif errorCode >= 100 and errorCode < 1100:
            sys.stderr.write("TWS ERROR - %s: %s\n" % (errorCode, errorString))
        elif errorCode >= 1100 and errorCode < 2100:
            sys.stderr.write("TWS SYSTEM-ERROR - %s: %s\n" % (errorCode, errorString))
        elif errorCode in (2104, 2106, 2108):
            sys.stderr.write("TWS INFO - %s: %s\n" % (errorCode, errorString))
        elif errorCode >= 2100 and errorCode <= 2110:
            sys.stderr.write("TWS WARNING - %s: %s\n" % (errorCode, errorString))
        else:
            sys.stderr.write("TWS ERROR - %s: %s\n" % (errorCode, errorString))
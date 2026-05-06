def getIPString():
        """ return comma delimited string of all the system IPs"""
        if not(NetInfo.systemip):
            NetInfo.systemip = ",".join(NetInfo.getSystemIps())
        return NetInfo.systemip
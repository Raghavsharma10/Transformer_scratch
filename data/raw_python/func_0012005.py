def get_remote_executors(hub_ip, port = 4444):
        ''' Get remote hosts from Selenium Grid Hub Console
        @param hub_ip: hub ip of selenium grid hub
        @param port: hub port of selenium grid hub
        '''
        resp = requests.get("http://%s:%s/grid/console" %(hub_ip, port))
        
        remote_hosts = ()
        if resp.status_code == 200:
            remote_hosts = re.findall("remoteHost: ([\w/\.:]+)",resp.text)
        return [host + "/wd/hub" for host in remote_hosts]
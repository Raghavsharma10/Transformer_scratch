def get_servers(self):
        """
        Create the list of Server object inside the Datacenter objects.
        Build an internal list of VM Objects (pro or smart) as iterator.
        :return: bool
        """
        json_scheme = self.gen_def_json_scheme('GetServers')
        json_obj = self.call_method_post(method='GetServers', json_scheme=json_scheme)
        self.json_servers = json_obj
        # if this method is called I assume that i must re-read the data
        # so i reinitialize the vmlist
        self.vmlist = VMList()
        # getting all instanced IP in case the list is empty
        if len(self.iplist) <= 0:
            self.get_ip()
        for elem in dict(json_obj)["Value"]:
            if elem['HypervisorType'] is 4:
                s = Smart(interface=self, sid=elem['ServerId'])
            else:
                s = Pro(interface=self, sid=elem['ServerId'])
            s.vm_name = elem['Name']
            s.cpu_qty = elem['CPUQuantity']
            s.ram_qty = elem['RAMQuantity']
            s.status = elem['ServerStatus']
            s.datacenter_id = elem['DatacenterId']
            s.wcf_baseurl = self.wcf_baseurl
            s.auth = self.auth
            s.hd_qty = elem['HDQuantity']
            s.hd_total_size = elem['HDTotalSize']
            if elem['HypervisorType'] is 4:
                ssd = self.get_server_detail(elem['ServerId'])
                try:
                    s.ip_addr = str(ssd['EasyCloudIPAddress']['Value'])
                except TypeError:
                    s.ip_addr = 'Not retrieved.'
            else:
                s.ip_addr = []
                for ip in self.iplist:
                    if ip.serverid == s.sid:
                        s.ip_addr.append(ip)
            self.vmlist.append(s)
        return True if json_obj['Success'] is True else False
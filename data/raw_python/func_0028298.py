def nextfreeip(self):
        """
        Method searches for the next free ip address in the scope object and returns it as a str
        value.
        :return:
        """
        allocated_ips = [ipaddress.ip_address(host['ip']) for host in self.hosts]
        for ip in self.netaddr:
            if str(ip).split('.')[-1] == '0':
                continue
            if ip not in allocated_ips:
                return ip
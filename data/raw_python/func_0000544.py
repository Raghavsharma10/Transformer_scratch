def add_chassis(self, chassis, port=22611, password='xena'):
        """ Add chassis.

        XenaManager-2G -> Add Chassis.

        :param chassis: chassis IP address
        :param port: chassis port number
        :param password: chassis password
        :return: newly created chassis
        :rtype: xenamanager.xena_app.XenaChassis
        """

        if chassis not in self.chassis_list:
            try:
                XenaChassis(self, chassis, port, password)
            except Exception as error:
                self.objects.pop('{}/{}'.format(self.owner, chassis))
                raise error
        return self.chassis_list[chassis]
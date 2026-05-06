def create(self, healthCheckNotification, instance, ipAddressResourceId, name, notificationContacts, rules,
               loadBalancerClassOfServiceID=1, *args, **kwargs):
        """
        :type healthCheckNotification: bool
        :type instance: list[Instance]
        :type ipAddressResourceId: list[int]
        :type loadBalancerClassOfServiceID: int
        :type name: str
        :type notificationContacts: NotificationContacts or list[NotificationContact]
        :type rules: Rules
        :param healthCheckNotification: Enable or disable notifications
        :param instance: List of balanced IP Addresses (VM or server)
        :param ipAddressResourceId: ID of the IP Address resource of the Load Balancer
        :param loadBalancerClassOfServiceID: default 1
        :param name: Name of the Load Balancer
        :param notificationContacts: Nullable if notificationContacts is false
        :param rules: List of NewLoadBalancerRule object containing the list of rules to be configured with the service
        """
        response = self._call(method=SetEnqueueLoadBalancerCreation,
                              healthCheckNotification=healthCheckNotification,
                              instance=instance,
                              ipAddressResourceId=ipAddressResourceId,
                              name=name,
                              notificationContacts=notificationContacts,
                              rules=rules,
                              loadBalancerClassOfServiceID=loadBalancerClassOfServiceID,
                              *args, **kwargs)
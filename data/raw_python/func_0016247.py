def get_notifications(self, startDate, endDate, loadBalancerID, loadBalancerRuleID):
        """
        Get the load balancer notifications for a specific rule within a specifying window time frame
        :type startDate: datetime
        :type endDate: datetime
        :type loadBalancerID: int
        :type loadBalancerRuleID: int
        :param startDate: From Date
        :param endDate: To Date
        :param loadBalancerID: ID of the Laod Balancer
        :param loadBalancerRuleID: ID of the Load Balancer Rule
        """
        return self._call(GetLoadBalancerNotifications, startDate=startDate, endDate=endDate,
                          loadBalancerID=loadBalancerID, loadBalancerRuleID=loadBalancerRuleID)
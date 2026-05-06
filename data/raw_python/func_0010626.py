def from_celery(cls, broker_dict):
        """ Create a BrokerStats object from the dictionary returned by celery.

        Args:
            broker_dict (dict): The dictionary as returned by celery.

        Returns:
            BrokerStats: A fully initialized BrokerStats object.
        """
        return BrokerStats(
            hostname=broker_dict['hostname'],
            port=broker_dict['port'],
            transport=broker_dict['transport'],
            virtual_host=broker_dict['virtual_host']
        )
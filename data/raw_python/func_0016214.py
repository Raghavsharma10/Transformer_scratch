def purchase_iscsi(self, quantity, iqn, name, protocol=SharedStorageProtocolType.ISCSI):
        """
        :type quantity: int
        :type iqn: list[str]
        :type name: str
        :type protocol: SharedStorageProtocols
        :param quantity: Amount of GB
        :param iqn: List of IQN represented in string format
        :param name: Name of the resource
        :param protocol: Protocol to use
        :return:
        """
        iqns = []
        for _iqn in iqn:
            iqns.append(SharedStorageIQN(Value=_iqn))
        request = self._call(SetEnqueuePurchaseSharedStorage, Quantity=quantity, SharedStorageName=name,
                             SharedStorageIQNs=iqns, SharedStorageProtocolType=protocol)
        response = request.commit()
        return response['Value']
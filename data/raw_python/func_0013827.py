def update_initiators(self, iqns=None, wwns=None):
        """Primarily for puppet-unity use.

        Update the iSCSI and FC initiators if needed.
        """
        # First get current iqns
        iqns = set(iqns) if iqns else set()
        current_iqns = set()
        if self.iscsi_host_initiators:
            current_iqns = {initiator.initiator_id
                            for initiator in self.iscsi_host_initiators}
        # Then get current wwns
        wwns = set(wwns) if wwns else set()
        current_wwns = set()
        if self.fc_host_initiators:
            current_wwns = {initiator.initiator_id
                            for initiator in self.fc_host_initiators}
        updater = UnityHostInitiatorUpdater(
            self, current_iqns | current_wwns, iqns | wwns)
        return updater.update()
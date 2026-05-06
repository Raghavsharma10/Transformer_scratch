def packets(self):
        """
        :return: dictionary {id: object} of all packets.
        :rtype: dict of (int, xenamanager.xena_port.XenaCapturePacket)
        """

        if not self.get_object_by_type('cappacket'):
            for index in range(0, self.read_stats()['packets']):
                XenaCapturePacket(parent=self, index='{}/{}'.format(self.index, index))
        return {p.id: p for p in self.get_objects_by_type('cappacket')}
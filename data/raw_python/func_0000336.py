def streams(self):
        """
        :return: dictionary {id: object} of all streams.
        :rtype: dict of (int, xenamanager.xena_stream.XenaStream)
        """

        if not self.get_objects_by_type('stream'):
            tpld_ids = []
            for index in self.get_attribute('ps_indices').split():
                stream = XenaStream(parent=self, index='{}/{}'.format(self.index, index))
                tpld_ids.append(stream.get_attribute('ps_tpldid'))
            if tpld_ids:
                XenaStream.next_tpld_id = max([XenaStream.next_tpld_id] + [int(t) for t in tpld_ids]) + 1
        return {s.id: s for s in self.get_objects_by_type('stream')}
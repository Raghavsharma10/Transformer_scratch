def add_stream(self, name=None, tpld_id=None, state=XenaStreamState.enabled):
        """ Add stream.

        :param name: stream description.
        :param tpld_id: TPLD ID. If None the a unique value will be set.
        :param state: new stream state.
        :type state: xenamanager.xena_stream.XenaStreamState
        :return: newly created stream.
        :rtype: xenamanager.xena_stream.XenaStream
        """

        stream = XenaStream(parent=self, index='{}/{}'.format(self.index, len(self.streams)), name=name)
        stream._create()
        tpld_id = tpld_id if tpld_id else XenaStream.next_tpld_id
        stream.set_attributes(ps_comment='"{}"'.format(stream.name), ps_tpldid=tpld_id)
        XenaStream.next_tpld_id = max(XenaStream.next_tpld_id + 1, tpld_id + 1)
        stream.set_state(state)
        return stream
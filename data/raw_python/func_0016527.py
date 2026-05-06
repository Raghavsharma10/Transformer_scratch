def qos(self, prefetch_size=0, prefetch_count=0, apply_global=False):
        """Request specific Quality of Service.

        This method requests a specific quality of service.  The QoS
        can be specified for the current channel or for all channels
        on the connection.  The particular properties and semantics of
        a qos method always depend on the content class semantics.
        Though the qos method could in principle apply to both peers,
        it is currently meaningful only for the server.

        :param prefetch_size: Prefetch window in octets.
            The client can request that messages be sent in
            advance so that when the client finishes processing a
            message, the following message is already held
            locally, rather than needing to be sent down the
            channel.  Prefetching gives a performance improvement.
            This field specifies the prefetch window size in
            octets.  The server will send a message in advance if
            it is equal to or smaller in size than the available
            prefetch size (and also falls into other prefetch
            limits). May be set to zero, meaning "no specific
            limit", although other prefetch limits may still
            apply. The ``prefetch_size`` is ignored if the
            :attr:`no_ack` option is set.

        :param prefetch_count: Specifies a prefetch window in terms of whole
            messages. This field may be used in combination with
            ``prefetch_size``; A message will only be sent
            in advance if both prefetch windows (and those at the
            channel and connection level) allow it. The prefetch-
            count is ignored if the :attr:`no_ack` option is set.

        :keyword apply_global: By default the QoS settings apply to the
            current channel only. If this is set, they are applied
            to the entire connection.

        """
        return self.backend.qos(prefetch_size, prefetch_count, apply_global)
def set_server_params(
            self, client_notify_address=None, mountpoints_depth=None, require_vassal=None,
            tolerance=None, tolerance_inactive=None, key_dot_split=None):
        """Sets subscription server related params.

        :param str|unicode client_notify_address: Set the notification socket for subscriptions.
            When you subscribe to a server, you can ask it to "acknowledge" the acceptance of your request.
            pointing address (Unix socket or UDP), on which your instance will bind and
            the subscription server will send acknowledgements to.

        :param int mountpoints_depth: Enable support of mountpoints of certain depth for subscription system.

            * http://uwsgi-docs.readthedocs.io/en/latest/SubscriptionServer.html#mountpoints-uwsgi-2-1

        :param bool require_vassal: Require a vassal field (see ``subscribe``) from each subscription.

        :param int tolerance: Subscription reclaim tolerance (seconds).

        :param int tolerance_inactive: Subscription inactivity tolerance (seconds).

        :param bool key_dot_split: Try to fallback to the next part in (dot based) subscription key.
            Used, for example, in SNI.

        """
        # todo notify-socket (fallback) relation
        self._set('subscription-notify-socket', client_notify_address)
        self._set('subscription-mountpoint', mountpoints_depth)
        self._set('subscription-vassal-required', require_vassal, cast=bool)
        self._set('subscription-tolerance', tolerance)
        self._set('subscription-tolerance-inactive', tolerance_inactive)
        self._set('subscription-dotsplit', key_dot_split, cast=bool)

        return self._section
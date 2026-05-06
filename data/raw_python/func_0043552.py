def _mapper(self):
        """
        Maps payment attributes to their specific types.

        :see :func:`~APIResource._mapper`
        """
        return {
            'card': Payment.Card,
            'customer': Payment.Customer,
            'hosted_payment': Payment.HostedPayment,
            'notification': Payment.Notification,
            'failure': Payment.Failure,
        }
def refund_payment(self):
        """
        Refund the payment using Stripe's refunding API.
        """
        Money = MoneyMaker(self.currency)
        filter_kwargs = {
            'transaction_id__startswith': 'ch_',
            'payment_method': StripePayment.namespace,
        }
        for payment in self.orderpayment_set.filter(**filter_kwargs):
            refund = stripe.Refund.create(charge=payment.transaction_id)
            if refund['status'] == 'succeeded':
                amount = Money(refund['amount']) / Money.subunits
                OrderPayment.objects.create(order=self, amount=-amount, transaction_id=refund['id'],
                                            payment_method=StripePayment.namespace)

        del self.amount_paid  # to invalidate the cache
        if self.amount_paid:
            # proceed with other payment service providers
            super(OrderWorkflowMixin, self).refund_payment()
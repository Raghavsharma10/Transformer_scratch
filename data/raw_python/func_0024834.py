def charge(self, cart, request):
        """
        Use the Stripe token from the request and charge immediately.
        This view is invoked by the Javascript function `scope.charge()` delivered
        by `get_payment_request`.
        """
        token_id = cart.extra['payment_extra_data']['token_id']
        if LooseVersion(SHOP_VERSION) < LooseVersion('0.11'):
            charge = stripe.Charge.create(
                amount=cart.total.as_integer(),
                currency=cart.total.currency,
                source=token_id,
                description=settings.SHOP_STRIPE['PURCHASE_DESCRIPTION']
            )
            if charge['status'] == 'succeeded':
                order = OrderModel.objects.create_from_cart(cart, request)
                order.add_stripe_payment(charge)
                order.save()
        else:
            order = OrderModel.objects.create_from_cart(cart, request)
            charge = stripe.Charge.create(
                amount=cart.total.as_integer(),
                currency=cart.total.currency,
                source=token_id,
                transfer_group=order.get_number(),
                description=settings.SHOP_STRIPE['PURCHASE_DESCRIPTION'],
            )
            if charge['status'] == 'succeeded':
                order.populate_from_cart(cart, request)
                order.add_stripe_payment(charge)
                order.save()

        if charge['status'] != 'succeeded':
            msg = "Stripe returned status '{status}' for id: {id}"
            raise stripe.error.InvalidRequestError(msg.format(**charge))
def get_payment_request(self, cart, request):
        """
        From the given request, add a snippet to the page.
        """
        try:
            self.charge(cart, request)
            thank_you_url = OrderModel.objects.get_latest_url()
            js_expression = 'window.location.href="{}";'.format(thank_you_url)
            return js_expression
        except (KeyError, stripe.error.StripeError) as err:
            raise ValidationError(err)
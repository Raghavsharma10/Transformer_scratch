def sync(self):
        """Overwrite local customer profile data with remote data"""
        output = get_profile(self.profile_id)
        output['response'].raise_if_error()
        for payment_profile in output['payment_profiles']:
            instance, created = CustomerPaymentProfile.objects.get_or_create(
                customer_profile=self,
                payment_profile_id=payment_profile['payment_profile_id']
            )
            instance.sync(payment_profile)
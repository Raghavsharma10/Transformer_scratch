def push_to_server(self):
        """
        Use appropriate CIM API call to save payment profile to Authorize.NET
        1. If customer has no profile yet, create one with this payment profile
        2. If payment profile is not on Authorize.NET yet, create it there
        3. If payment profile exists on Authorize.NET update it there
        """
        if not self.customer_profile_id:
            try:
                self.customer_profile = CustomerProfile.objects.get(
                    customer=self.customer)
            except CustomerProfile.DoesNotExist:
                pass
        if self.payment_profile_id:
            response = update_payment_profile(
                self.customer_profile.profile_id,
                self.payment_profile_id,
                self.raw_data,
                self.raw_data,
            )
            response.raise_if_error()
        elif self.customer_profile_id:
            output = create_payment_profile(
                self.customer_profile.profile_id,
                self.raw_data,
                self.raw_data,
            )
            response = output['response']
            response.raise_if_error()
            self.payment_profile_id = output['payment_profile_id']
        else:
            output = add_profile(
                self.customer.id,
                self.raw_data,
                self.raw_data,
            )
            response = output['response']
            response.raise_if_error()
            self.customer_profile = CustomerProfile.objects.create(
                customer=self.customer,
                profile_id=output['profile_id'],
                sync=False,
            )
            self.payment_profile_id = output['payment_profile_ids'][0]
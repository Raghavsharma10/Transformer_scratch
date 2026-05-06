def delete(self):
        """Delete the customer payment profile remotely and locally"""
        response = delete_payment_profile(self.customer_profile.profile_id,
                                          self.payment_profile_id)
        response.raise_if_error()
        return super(CustomerPaymentProfile, self).delete()
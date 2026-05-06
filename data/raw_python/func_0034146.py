def delete(self):
        """Delete the customer profile remotely and locally"""
        response = delete_profile(self.profile_id)
        response.raise_if_error()
        super(CustomerProfile, self).delete()
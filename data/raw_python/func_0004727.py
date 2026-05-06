def update(cls, customer_id, **kwargs):
        """
        Static method defined to update paystack customer data by id.

        Args:
            customer_id: paystack customer id.
            first_name: customer's first name(optional).
            last_name: customer's last name(optional).
            email: customer's email address(optional).
            phone:customer's phone number(optional).

        Returns:
            Json data from paystack API.
        """
        return cls().requests.put('customer/{customer_id}'.format(**locals()),
                                  data=kwargs)
def rate_tip_ride(self,
        ride_id,
        rating,
        tip_amount=None,
        tip_currency=None,
        feedback=None
    ):
        """Provide a rating, tip or feedback for the specified ride.
        Params
            ride_id (str)
                The unique ID of the Ride Request.
            rating (int)
                An integer between 1 and 5
            tip_amount
                Optional integer amount greater than 0 in minor currency units e.g. 200 for $2
            tip_currency
                Optional 3-character currency code e.g. 'USD'
            feedback
                Optional feedback message
        Returns
            (Response)
                A Response object with successful status_code
                if rating was submitted.
        """
        args = {
            "rating": rating,
            "tip.amount": tip_amount,
            "tip.currency": tip_currency,
            "feedback": feedback,
        }

        endpoint = 'v1/rides/{}/rating'.format(ride_id)
        return self._api_call('PUT', endpoint, args=args)
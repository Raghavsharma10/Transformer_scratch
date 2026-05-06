def get_images_by_tail_number(self, tail_number, page=1, limit=100):
        """Fetch the images of a particular aircraft by its tail number.

        This method can be used to get the images of the aircraft. The images are in 3 sizes and you can use what suits your need.

        Args:
            tail_number (str): The tail number, e.g. VT-ANL
            page (int): Optional page number; for users who are on a plan with flightradar24 they can pass in higher page numbers to get more data
            limit (int): Optional limit on number of records returned

        Returns:
            A dict with the images of the aircraft in various sizes

        Example::

            from pyflightdata import FlightData
            f=FlightData()
            #optional login
            f.login(myemail,mypassword)
            f.get_images_by_flight_number('VT-ANL')
            f.get_images_by_flight_number('VT-ANL',page=1,limit=10)
        """
        url = REG_BASE.format(tail_number, str(self.AUTH_TOKEN), page, limit)
        return self._fr24.get_aircraft_image_data(url)
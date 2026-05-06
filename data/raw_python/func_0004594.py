def _get_addresses(self, address_data, retain_name=False):
        """
        Takes RFC-compliant email addresses in both terse (email only)
        and verbose (name + email) forms and returns a list of
        email address strings

        (TODO: breaking change that returns a tuple of (name, email) per string)
        """
        if retain_name:
            raise NotImplementedError(
                "Not yet implemented, but will need client-code changes too"
            )

        # We trust than an email address contains an "@" after
        # email.utils.getaddresses has done the hard work. If we wanted
        # to we could use a regex to check for greater email validity

        # NB: getaddresses expects a list, so ensure we feed it appropriately
        if isinstance(address_data, str):
            if "[" not in address_data:
                # Definitely turn these into a list
                # NB: this is pretty assumptive, but still prob OK
                address_data = [address_data]

        output = [x[1] for x in getaddresses(address_data) if "@" in x[1]]
        return output
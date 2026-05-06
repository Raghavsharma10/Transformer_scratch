def set_domain(self, domain):
        """Sets a domain.

        arg:    domain (string): the new domain
        raise:  InvalidArgument - ``domain`` is invalid
        raise:  NoAccess - ``domain`` cannot be modified
        raise:  NullArgument - ``domain`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        if self.get_domain_metadata().is_read_only():
            raise errors.NoAccess()
        if not self._is_valid_string(domain,
                                     self.get_domain_metadata()):
            raise errors.InvalidArgument()
        self._my_map['domain']['text'] = domain
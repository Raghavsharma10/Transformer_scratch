def get_readable(self):
        '''
        Gets human-readable representation of the url (as unicode string,
        IRI according RFC3987)
        '''
        query = (u'?' + u'&'.join(u'{}={}'.format(urlquote(k), urlquote(v))
                                  for k, v in six.iteritems(self.query))
                 if self.query else '')
        hash_part = (u'#' + self.fragment) if self.fragment is not None else u''

        path, query, hash_part = uri_to_iri_parts(self.path, query, hash_part)

        if self.host:
            port = u':' + self.port if self.port else u''
            return u''.join((self.scheme, '://', self.host, port, path, query, hash_part))
        else:
            return path + query + hash_part
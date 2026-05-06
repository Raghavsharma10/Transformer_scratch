def pack(self):
        """Called to create a STOMP message from the internal values.
        """
        headers = ''.join(
            ['%s:%s\n' % (f, v) for f, v in sorted(self.headers.items())]
        )
        stomp_message = "%s\n%s\n%s%s\n" % (self._cmd, headers, self.body, NULL)

#        import pprint
#        print "stomp_message: ", pprint.pprint(stomp_message)

        return stomp_message
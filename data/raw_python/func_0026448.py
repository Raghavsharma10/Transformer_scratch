def _parse(self, bus, data):
        """
        Called when a sensor sends a new raw data to this serial connector.

        The data is sanitized and sent to the registered protocol
        listeners as time/raw/bus sentence tuple.
        """

        sen_time = time.time()

        try:
            # Split up multiple sentences
            if isinstance(data, bytes):
                data = data.decode('ascii')

            dirtysentences = data.split("\n")
            sentences = [(sen_time, x) for x in dirtysentences if x]

            def unique(it):
                s = set()
                for el in it:
                    if el not in s:
                        s.add(el)
                        yield el
                    else:
                        # TODO: Make sure, this is not identical but new data
                        self.log("Duplicate sentence received: ", el,
                                 lvl=debug)

            sentences = list(unique(sentences))
            return sentences
        except Exception as e:
            self.log("Error during data unpacking: ", e, type(e), lvl=error,
                     exc=True)
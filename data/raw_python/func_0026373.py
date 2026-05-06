def _augment_book(self, uuid, event):
        """
        Checks if the newly created object is a book and only has an ISBN.
        If so, tries to fetch the book data off the internet.

        :param uuid: uuid of book to augment
        :param client: requesting client
        """
        try:
            if not isbnmeta:
                self.log(
                    "No isbntools found! Install it to get full "
                    "functionality!",
                    lvl=warn)
                return

            new_book = objectmodels['book'].find_one({'uuid': uuid})
            try:
                if len(new_book.isbn) != 0:

                    self.log('Got a lookup candidate: ', new_book._fields)

                    try:
                        meta = isbnmeta(
                            new_book.isbn,
                            service=self.config.isbnservice
                        )

                        mapping = libraryfieldmapping[
                            self.config.isbnservice
                        ]

                        new_meta = {}

                        for key in meta.keys():
                            if key in mapping:
                                if isinstance(mapping[key], tuple):
                                    name, conv = mapping[key]
                                    try:
                                        new_meta[name] = conv(meta[key])
                                    except ValueError:
                                        self.log(
                                            'Bad value from lookup:',
                                            name, conv, key
                                        )
                                else:
                                    new_meta[mapping[key]] = meta[key]

                        new_book.update(new_meta)
                        new_book.save()

                        self._notify_result(event, new_book)
                        self.log("Book successfully augmented from ",
                                 self.config.isbnservice)
                    except Exception as e:
                        self.log("Error during meta lookup: ", e, type(e),
                                 new_book.isbn, lvl=error, exc=True)
                        error_response = {
                            'component': 'hfos.alert.manager',
                            'action': 'notify',
                            'data': {
                                'type': 'error',
                                'message': 'Could not look up metadata, sorry:' + str(e)
                            }
                        }
                        self.log(event, event.client, pretty=True)
                        self.fireEvent(send(event.client.uuid, error_response))

            except Exception as e:
                self.log("Error during book update.", e, type(e),
                         exc=True, lvl=error)

        except Exception as e:
            self.log("Book creation notification error: ", uuid, e, type(e),
                     lvl=error, exc=True)
def assert_no_union(self, collection1, collection2,
                        failure_message='Expected no overlap between collections: "{}" and "{}"'):
        """
        Asserts that the union of two sets is empty (collections are unique)
        """
        assertion = lambda: len(set(collection1).intersection(set(collection2))) == 0
        failure_message = unicode(failure_message).format(collection1, collection2)
        self.webdriver_assert(assertion, failure_message)
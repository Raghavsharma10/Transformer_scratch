def load_objects(self, addr, num_bytes, ret_on_segv=False):
        """
        Load memory objects from paged memory.

        :param addr: Address to start loading.
        :param num_bytes: Number of bytes to load.
        :param bool ret_on_segv: True if you want load_bytes to return directly when a SIGSEV is triggered, otherwise
                                 a SimSegfaultError will be raised.
        :return: list of tuples of (addr, memory_object)
        :rtype: tuple
        """

        result = []
        end = addr + num_bytes
        for page_addr in self._containing_pages(addr, end):
            try:
                #print "Getting page %x" % (page_addr // self._page_size)
                page = self._get_page(page_addr // self._page_size)
                #print "... got it"
            except KeyError:
                #print "... missing"
                #print "... SEGV"
                # missing page
                if self.allow_segv:
                    if ret_on_segv:
                        break
                    raise SimSegfaultError(addr, 'read-miss')
                else:
                    continue

            if self.allow_segv and not page.concrete_permissions & DbgPage.PROT_READ:
                #print "... SEGV"
                if ret_on_segv:
                    break
                raise SimSegfaultError(addr, 'non-readable')
            result.extend(page.load_slice(self.state, addr, end))

        return result
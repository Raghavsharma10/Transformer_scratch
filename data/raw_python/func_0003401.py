async def wait_for_group(self, container, networkid, timeout = 120):
        """
        Wait for a VXLAN group to be created
        """
        if networkid in self._current_groups:
            return self._current_groups[networkid]
        else:
            if not self._connection.connected:
                raise ConnectionResetException
            groupchanged = VXLANGroupChanged.createMatcher(self._connection, networkid, VXLANGroupChanged.UPDATED)
            conn_down = self._connection.protocol.statematcher(self._connection)
            timeout_, ev, m = await container.wait_with_timeout(timeout, groupchanged, conn_down)
            if timeout_:
                raise ValueError('VXLAN group is still not created after a long time')
            elif m is conn_down:
                raise ConnectionResetException
            else:
                return ev.physicalportid
async def main(self):
        """
        Main coroutine
        """
        try:
            lastkeys = set()
            dataupdate = FlowUpdaterNotification.createMatcher(self, FlowUpdaterNotification.DATAUPDATED)
            startwalk = FlowUpdaterNotification.createMatcher(self, FlowUpdaterNotification.STARTWALK)
            self.subroutine(self._flowupdater(), False, '_flowupdateroutine')
            # Cache updated objects
            presave_update = set()
            while True:
                self._restartwalk = False
                presave_update.update(self._updatedset)
                self._updatedset.clear()
                _initialkeys = set(self._initialkeys)
                try:
                    walk_result = await call_api(self, 'objectdb', 'walk',
                                                        {'keys': self._initialkeys, 'walkerdict': self._walkerdict,
                                                         'requestid': (self._requstid, self._requestindex)})
                except Exception:
                    self._logger.warning("Flow updater %r walk step failed, conn = %r", self, self._connection,
                                         exc_info=True)
                    # Cleanup
                    await call_api(self, 'objectdb', 'unwatchall',
                                         {'requestid': (self._requstid, self._requestindex)})
                    await self.wait_with_timeout(2)
                    self._requestindex += 1
                if self._restartwalk:
                    continue
                if self._updatedset:
                    if any(v.getkey() in _initialkeys for v in self._updatedset):
                        # During walk, there are other initial keys that are updated
                        # To make sure we get the latest result, restart the walk
                        continue
                lastkeys = set(self._savedkeys)
                _savedkeys, _savedresult = walk_result
                removekeys = tuple(lastkeys.difference(_savedkeys))
                self.reset_initialkeys(_savedkeys, _savedresult)
                _initialkeys = set(self._initialkeys)
                if self._dataupdateroutine:
                    self.terminate(self._dataupdateroutine)
                # Start detecting updates
                self.subroutine(self._dataobject_update_detect(_initialkeys, _savedresult), False, "_dataupdateroutine")
                # Set the updates back (potentially merged with newly updated objects)
                self._updatedset.update(v for v in presave_update)
                presave_update.clear()
                await self.walkcomplete(_savedkeys, _savedresult)
                if removekeys:
                    await call_api(self, 'objectdb', 'munwatch', {'keys': removekeys,
                                                                  'requestid': (self._requstid, self._requestindex)})
                # Transfer updated objects to updatedset2 before a flow update notification
                # This helps to make `walkcomplete` executes before `updateflow`
                #
                # But notice that since there is only a single data object copy in all the program,
                # it is impossible to hide the change completely during `updateflow`
                self._updatedset2.update(self._updatedset)
                self._updatedset.clear()
                self._savedkeys = _savedkeys
                self._savedresult = _savedresult
                await self.wait_for_send(FlowUpdaterNotification(self, FlowUpdaterNotification.FLOWUPDATE))
                while not self._restartwalk:
                    if self._updatedset:
                        if any(v.getkey() in _initialkeys for v in self._updatedset):
                            break
                        else:
                            self._updatedset2.update(self._updatedset)
                            self._updatedset.clear()
                            self.scheduler.emergesend(FlowUpdaterNotification(self, FlowUpdaterNotification.FLOWUPDATE))
                    await M_(dataupdate, startwalk)
        except Exception:
            self._logger.exception("Flow updater %r stops update by an exception, conn = %r", self, self._connection)
            raise
        finally:
            self.subroutine(send_api(self, 'objectdb', 'unwatchall', {'requestid': (self._requstid, self._requestindex)}),
                            False)
            if self._flowupdateroutine:
                self.terminate(self._flowupdateroutine)
                self._flowupdateroutine = None
            if self._dataupdateroutine:
                self.terminate(self._dataupdateroutine)
                self._dataupdateroutine = None
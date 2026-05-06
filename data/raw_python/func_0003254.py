async def _flowupdater(self):
        """
        Coroutine calling `updateflow()`
        """
        lastresult = set(v for v in self._savedresult if v is not None and not v.isdeleted())
        flowupdate = FlowUpdaterNotification.createMatcher(self, FlowUpdaterNotification.FLOWUPDATE)
        while True:
            currentresult = [v for v in self._savedresult if v is not None and not v.isdeleted()]
            # Calculating differences
            additems = []
            updateditems = []
            updatedset2 = self._updatedset2
            for v in currentresult:
                if v not in lastresult:
                    additems.append(v)
                else:
                    lastresult.remove(v)
                    if v in updatedset2:
                        # Updated
                        updateditems.append(v)
            removeitems = lastresult
            self._updatedset2.clear()
            # Save current result for next difference
            lastresult = set(currentresult)
            if not additems and not removeitems and not updateditems:
                await flowupdate
                continue
            await self.updateflow(self._connection, set(additems), removeitems, set(updateditems))
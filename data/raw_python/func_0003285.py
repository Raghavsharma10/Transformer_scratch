async def action_handler(self):
        """
        Call vtep controller in sequence, merge mutiple calls if possible
        
        When a bind relationship is updated, we always send all logical ports to a logicalswitch,
        to make sure it recovers from some failed updates (so called idempotency). When multiple
        calls are pending, we only need to send the last of them.
        """
        bind_event = VtepControllerCall.createMatcher(self._conn)
        event_queue = []
        timeout_flag = [False]

        async def handle_action():
            while event_queue or timeout_flag[0]:
                events = event_queue[:]
                del event_queue[:]

                for e in events:
                    # every event must have physname , phyiname
                    # physname: physical switch name - must be same with OVSDB-VTEP switch
                    # phyiname: physical port name - must be same with the corresponding port
                    physname = e.physname
                    phyiname = e.phyiname
                    if e.type == VtepControllerCall.UNBINDALL:
                        # clear all other event info
                        self._store_event[(physname,phyiname)] = {"all":e}
                    elif e.type == VtepControllerCall.BIND:
                        # bind will combine bind event before
                        vlanid = e.vlanid
                        if (physname,phyiname) in self._store_event:
                            v = self._store_event[(physname,phyiname)]

                            if vlanid in v:
                                logicalports = e.logicalports
                                v.update({vlanid:(e.type,e.logicalnetworkid,e.vni,logicalports)})
                                self._store_event[(physname,phyiname)] = v
                            else:
                                # new bind info , no combind event
                                v.update({vlanid:(e.type,e.logicalnetworkid,e.vni,e.logicalports)})
                                self._store_event[(physname,phyiname)] = v
                        else:
                            self._store_event[(physname,phyiname)] = {vlanid:(e.type,e.logicalnetworkid,
                                                                              e.vni,e.logicalports)}

                    elif e.type == VtepControllerCall.UNBIND:

                        vlanid = e.vlanid

                        if (physname,phyiname) in self._store_event:
                            v = self._store_event[(physname,phyiname)]
                            v.update({vlanid:(e.type,e.logicalnetworkid)})
                            self._store_event[(physname,phyiname)] = v
                        else:
                            self._store_event[(physname,phyiname)] = {vlanid:(e.type,e.logicalnetworkid)}

                    else:
                        self._parent._logger.warning("catch error type event %r , ignore it", exc_info=True)
                        continue

                call = []
                target_name = "vtepcontroller"
                for k,v in self._store_event.items():
                    if "all" in v:
                        # send unbindall
                        call.append(self.api(self,target_name,"unbindphysicalport",
                                             {"physicalswitch": k[0], "physicalport": k[1]},
                                             timeout=10))
                        # unbindall , del it whatever
                        del v["all"]

                try:
                    await self.execute_all(call)
                except Exception:
                    self._parent._logger.warning("unbindall remove call failed", exc_info=True)

                for k,v in self._store_event.items():
                    for vlanid , e in dict(v).items():
                        if vlanid != "all":
                            if e[0] == VtepControllerCall.BIND:

                                params = {"physicalswitch": k[0],
                                            "physicalport": k[1],
                                            "vlanid": vlanid,
                                            "logicalnetwork": e[1],
                                            "vni":e[2],
                                            "logicalports": e[3]}

                                try:
                                    await self.api(self,target_name,"updatelogicalswitch",
                                                  params,timeout=10)
                                except Exception:
                                    self._parent._logger.warning("update logical switch error,try next %r",params, exc_info=True)
                                else:
                                    del self._store_event[k][vlanid]

                            elif e[0] == VtepControllerCall.UNBIND:

                                params = {"logicalnetwork":e[1],
                                                "physicalswitch":k[0],
                                                "physicalport":k[1],
                                                  "vlanid":vlanid}

                                try:
                                    await self.api(self,target_name,"unbindlogicalswitch",
                                                      params,timeout=10)
                                except Exception:
                                    self._parent._logger.warning("unbind logical switch error,try next %r",params, exc_info=True)
                                else:
                                    del self._store_event[k][vlanid]

                self._store_event = dict((k,v) for k,v in self._store_event.items() if v)

                if timeout_flag[0]:
                    timeout_flag[0] = False

        def append_event(event, matcher):
            event_queue.append(event)

        while True:
            timeout, ev, m = await self.wait_with_timeout(10, bind_event)

            if not timeout:
                event_queue.append(ev)
            else:
                timeout_flag[0] = True

            await self.with_callback(handle_action(), append_event, bind_event)
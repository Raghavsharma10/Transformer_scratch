def debugrequest(self, event):
        """Handler for client-side debug requests"""
        try:
            self.log("Event: ", event.__dict__, lvl=critical)

            if event.data == "storejson":
                self.log("Storing received object to /tmp", lvl=critical)
                fp = open('/tmp/hfosdebugger_' + str(
                    event.user.useruuid) + "_" + str(uuid4()), "w")
                json.dump(event.data, fp, indent=True)
                fp.close()
            if event.data == "memdebug":
                self.log("Memory hogs:", lvl=critical)
                objgraph.show_most_common_types(limit=20)
            if event.data == "growth":
                self.log("Memory growth since last call:", lvl=critical)
                objgraph.show_growth()
            if event.data == "graph":
                self._drawgraph()
            if event.data == "exception":
                class TestException(BaseException):
                    """Generic exception to test exception monitoring"""

                    pass

                raise TestException
            if event.data == "heap":
                self.log("Heap log:", self.heapy.heap(), lvl=critical)
            if event.data == "buildfrontend":
                self.log("Sending frontend build command")

                self.fireEvent(frontendbuildrequest(force=True), "setup")
            if event.data == "logtail":
                self.fireEvent(logtailrequest(event.user, None, None,
                                              event.client), "logger")
            if event.data == "trigger_anchorwatch":
                from hfos.anchor.anchorwatcher import cli_trigger_anchorwatch
                self.fireEvent(cli_trigger_anchorwatch())

        except Exception as e:
            self.log("Exception during debug handling:", e, type(e),
                     lvl=critical)
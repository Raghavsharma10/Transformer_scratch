def _station(self) -> str:
        """Extract station name."""
        return str(self.obj.SBRes.SBReq.Start.Station.HafasName.Text.pyval)
def current_time(self) -> datetime:
        """Extract current time."""
        _date = datetime.strptime(self.obj.SBRes.SBReq.StartT.get("date"), "%Y%m%d")
        _time = datetime.strptime(self.obj.SBRes.SBReq.StartT.get("time"), "%H:%M")
        return datetime.combine(_date.date(), _time.time())
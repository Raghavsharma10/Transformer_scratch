def name():
        "Get/view the name for the well known ID of a Projection"
        if self.wkid in projected:
            return projected[self.wkid]
        elif self.wkid in geographic:
            return geographic[self.wkid]
        else:
            raise KeyError("Not a known WKID.")
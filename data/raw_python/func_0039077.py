def bbox(self):
        "Return the envelope as a Bound Box string compatible with (bb) params"
        return ",".join(str(attr) for attr in 
                            (self.xmin, self.ymin, self.xmax, self.ymax))
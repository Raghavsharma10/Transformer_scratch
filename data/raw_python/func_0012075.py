def shutdown(self):
        """Perform cleanup! We're goin' down!!!"""
        for ware in self.middleware:
            ware.preshutdown()
            self._shutdown()
            ware.postshutdown()
def reset(self):
        """
        Reseting wrapped function
        """
        super(SinonSpy, self).unwrap()
        super(SinonSpy, self).wrap2spy()
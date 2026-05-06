def checksum(self):
        """ Returns an MD5 digest of the model.

        This can be used to easily identify whether two models have the
        same architecture.
        """
        
        m = md5()
        for hl in self.hidden_layers:
            m.update(str(hl.architecture))
        m.update(str(self.top_layer.architecture))
        return m.hexdigest()
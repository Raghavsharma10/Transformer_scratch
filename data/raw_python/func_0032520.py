def repeater(self, req, tag):
        """
        Render some UI for repeating our form.
        """
        repeater = inevow.IQ(self.docFactory).onePattern('repeater')
        return repeater.fillSlots(
            'object-description', self.parameter.modelObjectDescription)
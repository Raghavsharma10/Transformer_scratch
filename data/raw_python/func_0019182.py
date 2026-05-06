def methodorder(self):
        """A list containing all methods of all |Node| and |Element| objects
        that need to be processed during a simulation time step in the
        order they must be called."""
        funcs = []
        for node in self.nodes:
            if node.deploymode == 'oldsim':
                funcs.append(node.sequences.fastaccess.load_simdata)
            elif node.deploymode == 'obs':
                funcs.append(node.sequences.fastaccess.load_obsdata)
        for node in self.nodes:
            if node.deploymode != 'oldsim':
                funcs.append(node.reset)
        for device in self.deviceorder:
            if isinstance(device, devicetools.Element):
                funcs.append(device.model.doit)
        for element in self.elements:
            if element.senders:
                funcs.append(element.model.update_senders)
        for element in self.elements:
            if element.receivers:
                funcs.append(element.model.update_receivers)
        for element in self.elements:
            funcs.append(element.model.save_data)
        for node in self.nodes:
            if node.deploymode != 'oldsim':
                funcs.append(node.sequences.fastaccess.save_simdata)
        return funcs
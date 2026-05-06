def get_vm_status(self, vm_name, rgn):
        '''
        this will just return the status of the virtual machine
        sometime the status may be unknown as shown by the azure portal;
        in that case statuses[1] doesn't exist, hence retrying on IndexError
        also, it may take on the order of minutes for the status to become
        available so the decorator will bang on it forever
        '''
        rgn = rgn if rgn else self.resource_group
        return self.client.virtual_machines.get(
            rgn, vm_name).instance_view.statuses[1].display_status
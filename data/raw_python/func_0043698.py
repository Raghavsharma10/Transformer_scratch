def get_vm(self, resource_group_name, vm_name):
        '''
        you need to retry this just in case the credentials token expires,
        that's where the decorator comes in
        this will return all the data about the virtual machine
        '''
        return self.client.virtual_machines.get(
            resource_group_name, vm_name, expand='instanceView')
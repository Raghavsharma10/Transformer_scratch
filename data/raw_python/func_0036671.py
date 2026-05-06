def organisations(self):
        '''The organisations of this composition.'''
        class Org:
            def __init__(self, sdo_id, org_id, members, obj):
                self.sdo_id = sdo_id
                self.org_id = org_id
                self.members = members
                self.obj = obj

        with self._mutex:
            if not self._orgs:
                for org in self._obj.get_owned_organizations():
                    owner = org.get_owner()
                    if owner:
                        sdo_id = owner._narrow(SDOPackage.SDO).get_sdo_id()
                    else:
                        sdo_id = ''
                    org_id = org.get_organization_id()
                    members = [m.get_sdo_id() for m in org.get_members()]
                    self._orgs.append(Org(sdo_id, org_id, members, org))
        return self._orgs
def parent_organisations(self):
        '''The organisations this RTC belongs to.'''
        class ParentOrg:
            def __init__(self, sdo_id, org_id):
                self.sdo_id = sdo_id
                self.org_id = org_id

        with self._mutex:
            if not self._parent_orgs:
                for sdo in self._obj.get_organizations():
                    if not sdo:
                        continue
                    owner = sdo.get_owner()
                    if owner:
                        sdo_id = owner._narrow(SDOPackage.SDO).get_sdo_id()
                    else:
                        sdo_id = ''
                    org_id = sdo.get_organization_id()
                    self._parent_orgs.append(ParentOrg(sdo_id, org_id))
        return self._parent_orgs
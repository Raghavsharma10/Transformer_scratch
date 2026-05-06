def parent_org_sdo_ids(self):
        '''The SDO IDs of the compositions this RTC belongs to.'''
        return [sdo.get_owner()._narrow(SDOPackage.SDO).get_sdo_id() \
                for sdo in self._obj.get_organizations() if sdo]
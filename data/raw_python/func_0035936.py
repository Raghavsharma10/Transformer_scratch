def delete_orderrun(self, orderrun_id):
        """
        :param self: self
        :param orderrun_id: string ; 'good' return a good value ; 'bad' return a bad value
        :rtype: DKReturnCode
        """
        rc = DKReturnCode()
        if orderrun_id == 'good':
            rc.set(rc.DK_SUCCESS, None, None)
        else:
            rc.set(rc.DK_FAIL, 'ServingDeleteV2: unable to delete OrderRun')
        return rc
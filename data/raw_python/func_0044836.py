def filter_slaves(selfie, slaves):
        """
        Remove slaves that are in an ODOWN or SDOWN state
        also remove slaves that do not have 'ok' master-link-status
        """
        return [(s['ip'], s['port']) for s in slaves
                if not s['is_odown'] and
                not s['is_sdown'] and
                s['master-link-status'] == 'ok']
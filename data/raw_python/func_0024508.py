def safe_filename(self, otype, oid):
        """Santize obj name into fname and verify doesn't already exist"""
        permitted = set(['_', '-', '(', ')'])
        oid = ''.join([c for c in oid if c.isalnum() or c in permitted])
        while oid.find('--') != -1:
            oid = oid.replace('--', '-')
        ext = 'json'
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        fname = ''
        is_new = False
        while not is_new:
            oid_len = 255 - len('%s--%s.%s' % (otype, ts, ext))
            fname = '%s-%s-%s.%s' % (otype, oid[:oid_len], ts, ext)
            is_new = True
            if os.path.exists(fname):
                is_new = False
                ts += '-bck'
        return fname
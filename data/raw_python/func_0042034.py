def upload(self, baseurl, filename):
        """Upload filename to this work"""
        # Prof is really dirty, we need to re-get the project page before upload
        payload = {
            'id_projet': self.field
        }
        prof_session.post(baseurl+"/main.php", params=payload)
        # We also need to get the upload page...
        payload = {
            'id': int(self.work_id)
        }
        prof_session.get(baseurl+"/upload.php", params=payload)
        # Finally we can actually send
        payload = {
            'MAX_FILE_SIZE': 1000000
        }
        prof_session.post(baseurl+'/upload2.php', files={'fichier1': open(filename, 'rb')}, params=payload)
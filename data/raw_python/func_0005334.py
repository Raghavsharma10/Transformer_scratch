def delete(self):
        """Deletes the record.
        """
        res = requests.delete(url=self.record_url, headers=HEADERS, verify=False)
        #self.write_response_html_to_file(res,"bob_delete.html")
        if res.status_code == 204:
            #No content. Can't render json:
            return {}
        return res.json()
def save_catalog(self):
        """
        Saves the catalog data to given key
        Cancels if the cmd is cancel
        Notifies user with the process.
        """
        if self.input["cmd"] == 'save_catalog':
            try:
                edited_object = dict()
                for i in self.input["form"]["CatalogDatas"]:
                    edited_object[i["catalog_key"]] = {"en": i["en"], "tr": i["tr"]}

                newobj = fixture_bucket.get(self.input["object_key"])
                newobj.data = edited_object
                newobj.store()

                # notify user by passing notify in output object
                self.output["notify"] = "catalog: %s successfully updated." % self.input[
                    "object_key"]
            except:
                raise HTTPError(500, "Form object could not be saved")
        if self.input["cmd"] == 'cancel':
            self.output["notify"] = "catalog: %s canceled." % self.input["object_key"]
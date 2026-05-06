def _parse_pet_record(self, root):
        """
        Given a <pet> Element from a pet.get or pet.getRandom response, pluck
        out the pet record.

        :param lxml.etree._Element root: A <pet> tag Element.
        :rtype: dict
        :returns: An assembled pet record.
        """
        record = {
            "breeds": [],
            "photos": [],
            "options": [],
            "contact": {},
        }

        # These fields can just have their keys and text values copied
        # straight over to the dict record.
        straight_copy_fields = [
            "id", "shelterId", "shelterPetId", "name", "animal", "mix",
            "age", "sex", "size", "description", "status", "lastUpdate",
        ]

        for field in straight_copy_fields:
            # For each field, just take the tag name and the text value to
            # copy to the record as key/val.
            node = root.find(field)
            if node is None:
                print("SKIPPING %s" % field)
                continue
            record[field] = node.text

        # Pets can be of multiple breeds. Find all of the <breed> tags and
        # stuff their text (breed names) into the record.
        for breed in root.findall("breeds/breed"):
            record["breeds"].append(breed.text)

        # We'll deviate slightly from the XML format here, and simply append
        # each photo entry to the record's "photo" key.
        for photo in root.findall("media/photos/photo"):
            photo = {
                "id": photo.get("id"),
                "size": photo.get("size"),
                "url": photo.text,
            }
            record["photos"].append(photo)

        # Has shots, no cats, altered, etc.
        for option in root.findall("options/option"):
            record["options"].append(option.text)

        # <contact> tag has some sub-tags that can be straight copied over.
        contact = root.find("contact")
        if contact is not None:
            for field in contact:
                record["contact"][field.tag] = field.text

        # Parse lastUpdate so we have a useable datetime.datime object.
        record["lastUpdate"] = self._parse_datetime_str(record["lastUpdate"])

        return record
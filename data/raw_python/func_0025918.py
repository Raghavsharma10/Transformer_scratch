def _remap_tag_to_tag_id(cls, tag, new_data):
        '''Remaps a given changed field from tag to tag_id.'''
        try:
            value = new_data[tag]
        except:
            # If tag wasn't changed, just return
            return

        tag_id = tag + '_id'
        try:
            # Remap the ID change to the required field
            new_data[tag_id] = value['id']
        except:
            try:
                # Try and grab the id of the object
                new_data[tag_id] = value.id
            except AttributeError:
                # If the changes field is not a dict or object, just use whatever value was given
                new_data[tag_id] = value

        # Remove the tag from the changed data
        del new_data[tag]
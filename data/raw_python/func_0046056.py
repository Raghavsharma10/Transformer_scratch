def _update_object_map(self, obj_map):
        """stub"""
        creation_time = obj_map['creationTime']
        obj_map['creationTime'] = dict()
        obj_map['creationTime']['year'] = creation_time.year
        obj_map['creationTime']['month'] = creation_time.month
        obj_map['creationTime']['day'] = creation_time.day
        obj_map['creationTime']['hour'] = creation_time.hour
        obj_map['creationTime']['minute'] = creation_time.minute
        obj_map['creationTime']['second'] = creation_time.second
        obj_map['creationTime']['microsecond'] = creation_time.microsecond
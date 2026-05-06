def classify(cls, o):
        """Break an Identity name into parts, or describe the type of other
        forms.

        Break a name or object number into parts and classify them. Returns a named tuple
        that indicates which parts of input string are name components, object number and
        version number. Does not completely parse the name components.

        Also can handle Name, Identity and ObjectNumbers

        :param o: Input object to split

        """
        # from collections import namedtuple

        s = str(o)

        if o is None:
            raise ValueError("Input cannot be None")

        class IdentityParts(object):
            on = None
            name = None
            isa = None
            name = None
            vname = None
            sname = None
            name_parts = None
            version = None
            cache_key = None

        # namedtuple('IdentityParts', ['isa', 'name', 'name_parts','on','version', 'vspec'])
        ip = IdentityParts()

        if isinstance(o, (DatasetNumber, PartitionNumber)):
            ip.on = o
            ip.name = None
            ip.isa = type(ip.on)
            ip.name_parts = None

        elif isinstance(o, Name):
            ip.on = None
            ip.isa = type(o)
            ip.name = str(o)
            ip.name_parts = ip.name.split(Name.NAME_PART_SEP)

        elif '/' in s:
            # A cache key
            ip.cache_key = s.strip()
            ip.isa = str

        elif cls.OBJECT_NUMBER_SEP in s:
            # Must be a fqname
            ip.name, on_s = s.strip().split(cls.OBJECT_NUMBER_SEP)
            ip.on = ObjectNumber.parse(on_s)
            ip.name_parts = ip.name.split(Name.NAME_PART_SEP)
            ip.isa = type(ip.on)

        elif Name.NAME_PART_SEP in s:
            # Must be an sname or vname
            ip.name = s
            ip.on = None
            ip.name_parts = ip.name.split(Name.NAME_PART_SEP)
            ip.isa = Name

        else:
            # Probably an Object Number in string form
            ip.name = None
            ip.name_parts = None
            ip.on = ObjectNumber.parse(s.strip())
            ip.isa = type(ip.on)

        if ip.name_parts:
            last = ip.name_parts[-1]

            try:
                ip.version = sv.Version(last)
                ip.vname = ip.name
            except ValueError:
                try:
                    ip.version = sv.Spec(last)
                    ip.vname = None  # Specs aren't vnames you can query
                except ValueError:
                    pass

            if ip.version:
                ip.name_parts.pop()
                ip.sname = Name.NAME_PART_SEP.join(ip.name_parts)
            else:
                ip.sname = ip.name

        return ip
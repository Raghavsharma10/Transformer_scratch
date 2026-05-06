def deserialize(self, apic_frame):
        """Convert APIC frame into Image."""
        return Image(data=apic_frame.data, desc=apic_frame.desc,
                     type=apic_frame.type)
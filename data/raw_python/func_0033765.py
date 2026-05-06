def add_pegasus_profile(self, namespace, key, value):
    """
    Add a Pegasus profile to this job which will be written to the dax as
    <profile namespace="NAMESPACE" key="KEY">VALUE</profile>
    This can be used to add classads to particular jobs in the DAX
    @param namespace: A valid Pegasus namespace, e.g. condor.
    @param key: The name of the attribute.
    @param value: The value of the attribute.
    """
    self.__pegasus_profile.append((str(namespace),str(key),str(value)))
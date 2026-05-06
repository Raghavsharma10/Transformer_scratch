def proxy_property_directly(bind_attr, attrs):
    """
    为类添加代理属性
    """
    def cls_builder(cls):
        """
        添加到类
        """
        for attr_name in attrs:
            setattr(cls, attr_name, _make_proxy_property(bind_attr, attr_name))
        return cls
    return cls_builder
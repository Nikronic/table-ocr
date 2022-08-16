__all__ = [
    'ExtendedEnum', 
]

# helpers
from enum import EnumMeta
from enum import Enum

class OnAccess(EnumMeta):
    """
    runs a user-specified function whenever member is accessed
    """
    #
    def __getattribute__(cls, name):
        obj = super().__getattribute__(name)
        if isinstance(obj, Enum) and obj._on_access:
            obj._on_access()
        return obj
    #
    def __getitem__(cls, name):
        member = super().__getitem__(name)
        if member._on_access:
            member._on_access()
        return member
    #
    def __call__(cls, value, names=None, *, module=None, qualname=None, type=None, start=1):
        obj = super().__call__(value, names, module=module, qualname=qualname, type=type, start=start)
        if isinstance(obj, Enum) and obj._on_access:
            obj._on_access()
        return obj

class ExtendedEnum(Enum, metaclass=OnAccess):
    """Same as :class:`Enum` but runs a user-specified function whenever member is accessed.
    
    Example:
    ::

        class MyEnum(ExtendedEnum):
            A = 1
            B = 2, True  # or any literal value
            C = 3
        
        MyEnum.A  # runs as normal
        MyEnum.B  # runs user defined function
        MyEnum.C  # runs as normal
    
    """
    def __new__(cls, value, *args):
        member = object.__new__(cls)
        member._value_ = value
        member._args = args
        member._on_access = member.raise_exception if args else None
        return member
    
    def deprecate(self):
        args = (self.name, ) + self._args
        import warnings
        warnings.warn(
                "member %r is deprecated; %s" % args,
                DeprecationWarning,
                stacklevel=3,
                )

    def raise_exception(self):
        args = (self.name, ) + self._args
        raise ValueError(f'Member {args[0]} is not valid.')
        

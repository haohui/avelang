"""Compile-time invariant helpers for AveLang kernels."""


def tag(target, fn, name=None):
    """Bind a compile-time tag function to a tensor or tile value."""
    pass


def assert_tag_eq(lhs, rhs):
    """Assert that two values carry equivalent compile-time tags."""
    pass


def reset_tags(target):
    """Reset tracked tags on a shared-memory value back to bottom."""
    pass


import os
from contextlib import contextmanager


@contextmanager
def temp_env_var(key: str, value: str):
    """Context manager for temporarily setting an environment variable.

    This context manager ensures that environment variables are properly set and restored,
    even if an exception occurs during the execution of the code block.

    Args:
        key: Environment variable name to set
        value: Value to set the environment variable to

    Yields:
        None

    Example:
        >>> with temp_env_var("MY_VAR", "test_value"):
        ...     # MY_VAR is set to "test_value"
        ...     do_something()
        ... # MY_VAR is restored to its original value or removed if it didn't exist
    """
    original = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if original is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original


def demo():
    @contextmanager
    def ctx():
        print("preparing context")
        try:
            yield
        finally:
            print("cleaning up context")

    print("=" * 50)
    with ctx():
        print("do something within context")


def nested_context():
    def ctx_a():
        print("preparing ctx a")
        try:
            yield
        finally:
            print("cleaning up ctx a")

    @contextmanager
    def ctx_b():
        print("preparing ctx b")
        try:
            yield
        finally:
            print("cleaning up ctx b")

    @contextmanager
    def switch_ctx(ctx_type="a", enabled=True):
        if not enabled:
            yield
            return

        if ctx_type == "a":
            # generator delegation
            yield from ctx_a()
        else:
            # or use another context manager
            with ctx_b():
                yield

    print("=" * 50)
    with switch_ctx(ctx_type="a"), switch_ctx(ctx_type="b"):
        print(f"work within ctx")


def reuse_context():
    @contextmanager
    def simple_context():
        print("preparing context")
        try:
            yield
        finally:
            print("cleaning up context")

    print("=" * 50)
    ctx = simple_context()
    with ctx:
        print("working")

    try:
        # cannot reuse the same context manager instance
        with ctx:
            print("working")
    except Exception as e:
        print(f"Reuse context error: {e}")


if __name__ == "__main__":
    demo()
    # nested_context()
    # reuse_context()

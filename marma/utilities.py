
import functools
from pathlib import Path
import os
import pickle

def cache(filename: Path):
    """
    Cache the result of a function.

    Beware: The decorator is invariant to changes in the code or different arguments;
    the cache has to be removed manually.
    """
    def decorator(func):
        @functools.wraps(func)
        def inner(*args, **kwargs):
            if os.path.isfile(filename):
                with open(filename, "rb") as infile:
                    return pickle.load(infile)
            retval = func(*args, **kwargs)

            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "wb") as outfile:
                pickle.dump(retval, outfile)

            return retval
        return inner
    return decorator


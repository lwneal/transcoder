from contextlib import contextmanager
import os

@contextmanager
def pushd(newDir):
    previousDir = os.getcwd()
    os.chdir(os.path.expanduser(newDir))
    yield
    os.chdir(previousDir)


from contextlib import contextmanager
import os
import sys
import unicodedata
import re


@contextmanager
def pushd(newDir):
    previousDir = os.getcwd()
    os.chdir(os.path.expanduser(newDir))
    yield
    os.chdir(previousDir)


def chdir_to_experiment(experiment_name):
    name = slugify(unicode(experiment_name))
    os.chdir(os.path.expanduser('~/results'))
    if not os.path.exists(name):
        os.mkdir(name)
    os.chdir(name)


def slugify(value):
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.
    """
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore')
    value = unicode(re.sub('[^\w\s-]', '', value).strip().lower())
    return unicode(re.sub('[-\s]+', '-', value))


def redirect_stdout_stderr(name):
    class Logger(object):
        def __init__(self, name='stdout.log'):
            self.terminal = sys.stdout
            self.log = open(name, "a")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

    if name:
        sys.stdout = Logger(name)

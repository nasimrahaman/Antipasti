"""General Tools"""

import yaml
import os


class FileVariable(object):
    """Class to encapsulate variables in a yaml file."""
    def __init__(self, ymlfile, key):
        # Meta
        self.ymlfile, self.key = ymlfile, key
        # Init last modified
        self.lastmodified = 0.
        # Init last known value
        self.lastknownvalue = None

    def get_value(self):
        # Check if there are changes to the read
        filehaschanged = os.stat(self.ymlfile).st_mtime != self.lastmodified

        # Update lastmodified timestamp
        if filehaschanged:
            self.lastmodified = os.stat(self.ymlfile).st_mtime
        else:
            # Return the last known value
            return self.lastknownvalue

        # If the file has been modified, read from it
        with open(self.ymlfile, 'r') as f:
            update = yaml.load(f)

        # Read from update and return
        out = self.lastknownvalue = update.get(self.key, self.lastknownvalue)
        return out


import sys

class Logger:
    def __init__(self, opath="stderr", 
                 indent_level=0, indentation="\t", 
                 ):
        self._opath = opath
        self._indent_level = indent_level
        self._indentation = indentation
    
    def indent(self):
        '''
        Increase the indent level.
        '''
        self._indent_level += 1

    def unindent(self):
        '''
        Decrease the indent level.
        '''
        if self._indent_level > 0:
            self._indent_level -= 1
        else:
            raise ValueError("Indent level cannot be negative.")

    def take_log(self, message):
        '''
        Take log according to the indent level set.
        '''
        self._write(
            self._indentation * self._indent_level + message
        )

    def _write(self, message):
        '''
        Write message to the logging stream. 
        '''
        if self._opath == "stderr":
            sys.stderr.write(message)
            sys.stderr.write("\n")

        elif self._opath == "stdout":
            sys.stdout.write(message)
            sys.stdout.write("\n")

        else:
            with open(self._opath, "a") as f:
                f.write(message)
                f.write("\n")

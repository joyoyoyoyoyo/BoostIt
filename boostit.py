import sys

class Context():
    def __init__(self, usage):
        if len(usage) != 6:
            print 'Error: usage is <ensemble size> <pos-train-file> <neg-train-file> <pos-test-file> <neg-test-file>\n\nGiven input:\t{0}'.format(usage)
        self._ensemble_size = usage[1]
        self.training_files = [usage[2], usage[3]]
        self.testing_files = [usage[4], usage[5]]
                 #setattr(self, name, value)


if __name__ == '__main__':
    ctx = Context(sys.argv)
    
    


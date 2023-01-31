from abc import ABC, abstractmethod

class TestsuiteProcessor(ABC):
    @abstractmethod
    def write_output(self, dest_path, data):
        pass

    @abstractmethod
    def process_testsuites(self, treebanks_path, lextypes):
        pass

    @abstractmethod
    def process_one_testsuite(self, tsuite, type, lextypes):
        pass

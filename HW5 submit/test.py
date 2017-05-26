"""
Classic MapReduce job: count the frequency of words.
"""
"""
from mrjob.job import MRJob
import re

WORD_RE = re.compile(r"[\w']+")


class MRWordFreqCount(MRJob):

    def mapper(self, _, line):
        for word in WORD_RE.findall(line):
            yield (word.lower(), 1)

    def combiner(self, word, counts):
        yield (word, sum(counts))

    def reducer(self, word, counts):
        yield (word, sum(counts))


if __name__ == '__main__':
     MRWordFreqCount.run()
"""

import subprocess

file = subprocess.Popen(["hadoop", "fs", "-cat", "./mapper_input_file.txt"], stdout=subprocess.PIPE)
for line in file.stdout:
    print line
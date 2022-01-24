

class ERG_Corpus:
    def __init__(self,name,vectors,labels, sen_lengths):
        self.name = name
        self.X = vectors
        self.Y = labels
        self.sen_lengths = sen_lengths


'''
Assuming the following tag-separated format:
VBP+RB	VBP
VBZ+RB	VBZ
IN+DT	IN
(etc.)
'''

class Pos_mapper:
    def __init__(self, filepath):
        with open(filepath,'r') as f:
            lines = f.readlines()
        self.pos_map = {}
        self.unknowns = []
        for ln in lines:
            if ln:
                tag,mapping = ln.strip().split('\t')
                self.pos_map[tag] = mapping

    def map_tag(self,tag):
        if tag in self.pos_map:
            return self.pos_map[tag]
        else:
            #return the first tag
            self.unknowns.append(tag)
            #print('Unknown POS tag: ' + tag)
            tags = tag.split('+')
            return tags[0]
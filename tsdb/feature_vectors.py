'''
These methods are for processing pydelphin-style itsdb test suites,
preparing the data for token classification based on feature vectors for scikit-learn ML library.

The output is binary files which are loadable with scikit-learn.
'''
import pickle
from delphin import itsdb
from delphin.tokens import YYTokenLattice
from tsdb.TestsuiteProcessor import ProcessedCorpus, TestsuiteProcessor
from tsdb import pos_map

EOS_TOKEN = 'EOS'
CONTEXT_WINDOW = 2

class Feature_Vec_Extractor(TestsuiteProcessor):

    def __init__(self):
        pass

    @property
    def output_format(self):
        return 'wb'

    def process_one_testsuite(self, tsuite, type, lextypes):
        ts = itsdb.TestSuite(tsuite)
        all_sentences = []
        parsed_sentences = []
        items = list(ts.processed_items())
        observations = []
        total_tokens = 0
        print("{} sentences in corpus {} including possible sentences with no parse.".format(len(items), ts.path.stem))
        for response in items:
            all_sentences.append(response['i-input'])
            if len(response['results']) > 0:
                parsed_sentences.append(response['i-input'])
                deriv = response.result(0).derivation()
                p_input = response['p-input']
                p_tokens = response['p-tokens']
                if p_tokens:
                    terminals_tok_tags = self.map_lattice_to_input(p_input, p_tokens, deriv)
                    is_test_data = type=='test'
                    observations.append(self.get_observations(terminals_tok_tags, lextypes, is_test_data))
                else:
                    print("Skipping a sentence from corpus {} because it does not have p-tokens.".format(ts.path.stem))
        pc = ProcessedCorpus(ts.path.stem, type, observations, all_sentences, parsed_sentences, total_tokens )
        return pc

    def map_lattice_to_input(self, p_input, p_tokens, deriv):
        terminals_toks_postags = []
        yy_lattice = YYTokenLattice.from_string(p_tokens)
        yy_input = YYTokenLattice.from_string(p_input)
        for t in deriv.terminals():
            toks_pos_tags = []
            for ttok in t.tokens:
                span = None
                pos_probs = {}
                for lat_tok in yy_lattice.tokens:
                    if lat_tok.id == ttok.id:
                        span = lat_tok.lnk.data
                        break
                for i,in_tok in enumerate(yy_input.tokens):
                    if in_tok.lnk.data[0] == span[0]:
                        for pos, p in in_tok.pos:
                            if pos not in pos_probs:
                                pos_probs[pos] = []
                            pos_probs[pos].append(float(p))
                        if in_tok.lnk.data[1] != span[1]:
                            cur_tok = in_tok
                            while cur_tok.lnk.data[1] != span[1]:
                                next_tok = yy_input.tokens[i+1]
                                i += 1
                                for pos, p in next_tok.pos:
                                    if pos not in pos_probs:
                                        pos_probs[pos] = []
                                    pos_probs[pos].append(float(p))
                                cur_tok = next_tok
                        else:
                            break
                toks_pos_tags.append((ttok, pos_probs))
            terminals_toks_postags.append((t,toks_pos_tags))
        return terminals_toks_postags

    '''
    Input: a list of pydelphin itsdb terminals and a list of known lexical types (e.g. from the training data).
    Output: a list of tuples: the terminal orthographic form and its lexical type, if it is known, or <UNK> otherwise.
    '''
    def get_observations(self, terminals_tok_tags, lextypes, is_test_data):
        pos_mapper = pos_map.Pos_mapper('pos-map.txt')
        x = []
        y = []
        tokens, labels, pos_tags = \
            self.get_tokens_labels(terminals_tok_tags, CONTEXT_WINDOW, lextypes, pos_mapper)
        for k, t in enumerate(tokens):
            if k < CONTEXT_WINDOW or k >= len(tokens) - CONTEXT_WINDOW:
                continue
            y.append(labels[k])
            # For non-autoregressive models, do not pass labels
            x.append(self.get_context(t, tokens, pos_tags, k, CONTEXT_WINDOW, is_test_data))
        return (x,y)

    def get_tokens_labels(self, terms_and_tokens_tags, context_window, lextypes, pos_mapper):
        tokens = []
        labels = []
        pos_tags = []
        for i,(terminal, toks_tags) in enumerate(terms_and_tokens_tags):
            letype = str(lextypes.get(terminal.parent.entity, "<UNK>"))
            tokens.append(terminal.form)
            labels.append(letype)
            pos_tags.append(self.get_pos_tag(toks_tags, pos_mapper))
        for i in range(1,1+context_window):
            tokens.insert(0, 'FAKE-' + str(i))
            labels.insert(0, 'FAKE-' + str(i))
            pos_tags.insert(0,'FAKE-' + str(i))
            tokens.append('FAKE+' + str(i))
            labels.append('FAKE+' + str(i))
            pos_tags.append('FAKE+' + str(i))
        return tokens, labels, pos_tags

    def get_pos_tag(self,tokens_tags, pos_mapper):
        tag = ''
        for tt in tokens_tags:
            pos_probs = tt[1]
            for pos in pos_probs:
                tag = tag + '+' + pos
        tag = tag.strip('+')
        if '+' in tag:
            tag = pos_mapper.map_tag(tag)
        return tag

    def get_context(self, t, tokens, pos_tags, i, window, is_test_data, labels=None):
        context = {'w': t, 'pos': pos_tags[i]}
        for j in range(1,window+1):
            prev_tok = tokens[i-j]
            prev_pos = pos_tags[i-j]
            next_tok = tokens[i+j]
            next_pos = pos_tags[i+j]
            context['w-' + str(j)] = prev_tok
            context['w+' + str(j)] = next_tok
            context['pos-' + str(j)] = prev_pos
            context['pos+' + str(j)] = next_pos
            if labels: # only in autoregressive models
                prev_tag = labels[i-j] if not is_test_data else None
                context['tag-' + str(j)] = prev_tag
        return context

    def get_output_for_one_corpus(self, pc, total_sen, total_tok):
        data = {'ft': [], 'lt': []}
        for x,y in pc.processed_data:
            data['ft'].extend(x)
            data['lt'].extend(y)
            total_tok += len(x)
            total_sen += 1
        return data, total_sen, total_tok

    def write_output_by_split(self, dest_path, data):
        print('Writing output to {}'.format(dest_path))
        for split_type in ['train', 'dev', 'test']:
            with open(dest_path + split_type + '/' + split_type, self.output_format) as f:
                whole_output = {'ft': [], 'lt': []}
                total_sen = 0
                total_tok = 0
                for pc in data[split_type]:
                    output, total_sen, total_tok = self.get_output_for_one_corpus(pc, total_sen, total_tok)
                    whole_output['ft'].extend(output['ft'])
                    whole_output['lt'].extend(output['lt'])
                pickle.dump(whole_output,f)
                print('Wrote {} sentences, {} tokens out for {}.'.format(total_sen, total_tok, split_type))

    def write_output_by_corpus(self, dest_path, data):
        print('Writing output to {}'.format(dest_path))
        # Training and dev data is lumped all together
        for split_type in ['train', 'dev']:
            with open(dest_path + split_type + '/' + split_type, self.output_format) as f:
                whole_output = {'ft': [], 'lt': []}
                total_sen = 0
                total_tok = 0
                for pc in data[split_type]:
                    output, total_sen, total_tok = self.get_output_for_one_corpus(pc, total_sen, total_tok)
                    whole_output['ft'].extend(output['ft'])
                    whole_output['lt'].extend(output['lt'])
                pickle.dump(whole_output,f)
                print('Wrote {} sentences, {} tokens out for {}.'.format(total_sen, total_tok, split_type))
        # Test data is kept separately by corpus, to be able to look at accuracy with different domains
        for pc in data['test']:
            with open(dest_path + 'test' + '/' + pc.name, self.output_format) as f:
                total_sen = 0
                total_tok = 0
                output, total_sen, total_tok = self.get_output_for_one_corpus(pc, total_sen, total_tok)
                pickle.dump(output,f)
                print('Wrote {} sentences, {} tokens out for {}.'.format(total_sen, total_tok, pc.name))

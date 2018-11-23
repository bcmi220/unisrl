import json
import sys

'''
We treat on sentence as a document
'''
def convert_sent_dict(doc_key, sentence):
    speakers = [['-' for _ in sentence]]
    sentences = [[line[1] for line in sentence]]
    constituents = [[]]
    clusters = []
    ner = [[]]
    srl = [[]]
    predicate_idx = 0
    for idx in range(len(sentence)):
        
        if sentence[idx][12] == 'Y': # is predicate
            no_arg = True
            for jdx in range(len(sentence)):
                if sentence[jdx][14+predicate_idx] != '_':
                    no_arg = False
                    srl[0].append([idx, jdx, jdx, sentence[jdx][14+predicate_idx]])
            predicate_idx += 1
            if no_arg: # in case of no argument
                srl[0].append([idx, idx, idx, '_'])
    return {
        'speakers':speakers,
        'doc_key':doc_key,
        'sentences':sentences,
        'srl':srl,
        'constituents':constituents,
        'clusters':clusters,
        'ner':ner
    }



def conll09_to_json(dataset_path, output_path):
    with open(dataset_path, 'r') as f:
        data = f.readlines()
    
    sentences = []
    sent = []
    for line in data:
        if len(line.strip())>0:
            sent.append(line.strip().split('\t'))
        else:
            if len(sent)>0:
                sentences.append(sent)
                sent = []
    
    if len(sent)>0:
        sentences.append(sent)

    with open(output_path, 'w') as f:
        for idx in range(len(sentences)):
            json_data = convert_sent_dict('S'+str(idx), sentences[idx])
            f.write(json.dumps(json_data)+'\n')

if __name__ == "__main__":
    conll09_to_json(sys.argv[1], sys.argv[2])

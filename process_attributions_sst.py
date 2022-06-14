import json
import numpy as np
import scipy.stats
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--attrs_dir', required=True, help='Output directory of the create_attributions_sst script')
parser.add_argument('--output_file', default='metrics.csv', help='File to write the results to')
parser.add_argument('--absolute', default=False, help='If True, absolute attributions, else Pos/Neg attributions')

args = parser.parse_args()

OUTPUT_FILE = args.output_file
np.random.seed(42)
ATTRS_DIR = args.attrs_dir
ATTRS_ABS = args.absolute
LOWERCASE_SST = True if 'uncased' in ATTRS_DIR else False

MINIMAL_TOKEN_COUNT = 12

accent_dict = {
    'á': 'a',
    'ć': 'c',
    'č': 'c',
    'ď': 'd',
    'é': 'e',
    'ě': 'e',
    'è': 'e',
    'í': 'i',
    'ľ': 'l',
    'ň': 'n',
    'ó': 'o',
    'ř': 'r',
    'š': 's',
    'ť': 't',
    'ú': 'u',
    'ů': 'u',
    'ü': 'u',
    'û': 'u',
    'ý': 'y',
    'ž': 'z',
}


def get_phrase_sentiments():
    return load_json('datasets_ours/sst/phrase_sentiments.json')


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.loads(f.read())


def remove_accents(string):
    new_string = ""
    for i in range(len(string)):
        if string[i] in accent_dict.keys():
            new_string += accent_dict[string[i]]
        else:
            new_string += string[i]
    return new_string


def get_method_file_dict():
    return load_json(ATTRS_DIR + '/method_file_dict_custom.json')


def get_tokens():
    return load_json(ATTRS_DIR + '/sst_bert_tokens.json')


def to_abs_values(values):
    return rec_abs(values)


def rec_abs(a):
    new_list = []
    for el in a:
        if isinstance(el, list):
            new_list.append(rec_abs(el))
        else:
            new_list.append(np.abs(el))
    return new_list


def generate_random_attrs(method_file_dict):
    # choose the first attrs file to get the dimensions of the attributions
    method = 'grads'
    attrs_file = method_file_dict[method]
    attrs = load_json(ATTRS_DIR + "/" + attrs_file.split('/')[-1])
    random_attrs = []
    random_time = 0

    for attr in attrs[0]:
        shape = np.array(attr).shape
        r = np.random.uniform(-0.5, 0.5, shape)
        random_attrs.append(r.tolist())

    res = [random_attrs, random_time]
    random_file = ATTRS_DIR + "/random.json"
    with open(random_file, 'w+', encoding='utf-8') as f:
        f.write(json.dumps(res))

    method_file_dict['random'] = 'attrs/random.json'


def get_short_sample_indices(sst_bert_tokens):
    sst_tokens = sst_bert_tokens['sst_tokens']
    print(len(sst_tokens))
    indices = []
    for i in range(len(sst_tokens)):
        if len(sst_tokens[i]) < MINIMAL_TOKEN_COUNT:
            indices.append(i)
    return indices


def lowercase_sst_tokens(sst_tokens):
    lowercased_all = []
    for tokens in sst_tokens:
        lowercased = []
        for token in tokens:
            lowercased.append(token.lower())
        lowercased_all.append(lowercased)
    return lowercased_all


def preprocess_token_attrs(sst_bert_tokens: dict, attributions: list):
    # take care of the differences between SST and BERT tokens
    bert_tokens = sst_bert_tokens['bert_tokens']
    sst_tokens = sst_bert_tokens['sst_tokens']

    # process each sample
    processed_attributions = []
    for sample_index in range(len(bert_tokens)):
        sample_attributions = []
        # if the two don't differ in length, the tokens match since we have
        # created the sentences by joining the tokens - that means that BERT
        # tokenizer can split the sst token but it can't merge any, thus
        # we can do this
        if len(sst_tokens[sample_index]) == len(bert_tokens[sample_index]):
            processed_attributions.append(attributions[0][sample_index])
            continue

        sst = sst_tokens[sample_index]
        bert = bert_tokens[sample_index]

        # now we iterate over the sst tokens, checking if the BERT tokens match the SST tokens
        # sst_tokens will be shorter or equal length compared to bert_tokens
        bert_index = 0
        for sst_index in range(len(sst)):
            if sst[sst_index] == bert[bert_index]:
                # if the tokens match, the attribution stays
                sample_attributions.append(attributions[0][sample_index][bert_index])
                bert_index += 1
            else:
                # otherwise we need to iterate forwards to match multiple bert tokens to one sst token
                temp = bert[bert_index]
                start = bert_index

                if not LOWERCASE_SST:
                    while temp != sst[sst_index]:
                        bert_index += 1
                        if bert[bert_index][0] == '#':
                            temp += bert[bert_index][2:]
                        else:
                            temp += bert[bert_index]
                    bert_index += 1
                else:
                    sst_temp = remove_accents(sst[sst_index])
                    while temp != sst_temp:
                        bert_index += 1
                        try:
                            if bert[bert_index][0] == '#':
                                temp += bert[bert_index][2:]
                            else:
                                temp += bert[bert_index]
                        except IndexError:
                            a = 0
                    bert_index += 1
                size = bert_index - start
                _sum = 0
                for i in range(size):
                    _sum += attributions[0][sample_index][start + i]
                sample_attributions.append(_sum / size)

        processed_attributions.append(sample_attributions)

    return processed_attributions, attributions[1]


def get_sst_sentiments(sst_tokens: list, phrase_sentiments: dict):
    # get sentiments for tokens
    sentiments = []
    for sentence in sst_tokens:
        s = []
        for token in sentence:
            s.append(phrase_sentiments[token])
        sentiments.append(s)

    return sentiments


def get_sst_labels(sst_bert_tokens: dict, phrase_sentiments: dict):
    labels = []
    for sentence in sst_bert_tokens['sst_tokens']:
        text = ""
        for i in range(len(sentence)):
            if i == len(sentence) - 1:
                text += sentence[i]
            else:
                text += sentence[i] + " "
        sentiment = phrase_sentiments[text]
        labels.append(int(round(sentiment)))

    return labels


def merge_embedding_attrs(attributions: list):
    # preprocess the embedding attributions
    # if the shape has only one dimension, there is nothing to do
    if len(np.array(attributions[0][0]).shape) == 1:
        return attributions

    # average/sum over embeddings
    attrs_processed = []
    for attr in attributions[0]:
        attr = np.array(attr)
        attr = np.average(attr, axis=1)
        # attr = np.sum(attr, axis=1)
        attrs_processed.append(attr.tolist())

    return [attrs_processed, attributions[1]]


def scale_shift_attrs(attributions: list):
    # scale the attributions to <-0.5, 0.5>,
    scaled_attrs = []
    for sentence_attrs in attributions[0]:
        sentence_attrs = np.array(sentence_attrs)
        _max = max([abs(x) for x in sentence_attrs])
        if _max != 0:
            sentence_attrs = np.array(sentence_attrs) / _max / 2
        scaled_attrs.append(list(sentence_attrs))

    return [scaled_attrs, attributions[1]]


def scale_sst_attrs(sst_attrs: list):
    # shift the center to 0 (neutral 0.5 to 0), then scale the attributions to <-0.5, 0.5>
    scaled_attrs = []
    for sentence_attrs in sst_attrs:
        sentence_attrs = np.array(sentence_attrs)
        sentence_attrs -= 0.5

        _max = max([abs(x) for x in sentence_attrs])
        if _max != 0:
            sentence_attrs = sentence_attrs / _max / 2
        scaled_attrs.append(list(sentence_attrs))

    return scaled_attrs


def get_max_k_indices(attrs, k):
    top_indices = []
    for j in range(k):
        _max = -2.0
        index = 0
        for i in range(len(attrs)):
            if attrs[i] > _max and i not in top_indices:
                _max = attrs[i]
                index = i

        top_indices.append(index)
    return top_indices


def get_min_k_indices(attrs, k):
    top_indices = []
    for j in range(k):
        _min = 2.0
        index = 0
        for i in range(len(attrs)):
            if attrs[i] < _min and i not in top_indices:
                _min = attrs[i]
                index = i

        top_indices.append(index)
    return top_indices


def eval_top_k(bert_attrs, sst_attrs, label, K):
    # compare top k
    if label == 1 or ATTRS_ABS:
        top_bert_indices = get_max_k_indices(bert_attrs, K)
        top_sst_indices = get_max_k_indices(sst_attrs, K)
    else:
        top_bert_indices = get_max_k_indices(bert_attrs, K)
        top_sst_indices = get_min_k_indices(sst_attrs, K)

    count = 0
    for i in top_bert_indices:
        if i in top_sst_indices:
            count += 1

    return count


def evaluate_attr(bert_attrs: list, sst_sentiments: list, label: int):
    # evaluate the attributions for one sentence
    res = {
        'top1': eval_top_k(bert_attrs, sst_sentiments, label, 1),
        'top3': eval_top_k(bert_attrs, sst_sentiments, label, 3) / 3.0,
        'top5': eval_top_k(bert_attrs, sst_sentiments, label, 5) / 5.0,
    }

    return res


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


def process_method(bert_attrs: list, sst_attrs: list, short_samples_indices: list, sst_bert_tokens: dict, labels: list):
    # perform preprocessing (average embedding attributions except for relprop)
    # we need to preprocess the sentiment too - we scale it to <0, 1>, so that the highest sentiment will have the
    # same value as the highest attribution
    bert_attrs = merge_embedding_attrs(bert_attrs)
    bert_attrs = scale_shift_attrs(bert_attrs)
    bert_attrs = preprocess_token_attrs(sst_bert_tokens, bert_attrs)

    if ATTRS_ABS:
        bert_attrs = to_abs_values(bert_attrs)

    # evaluate the preprocessed attributions
    evaluations = {
        'top1': [],
        'top3': [],
        'top5': []
    }

    i = 0
    for bert_attr, sst_attr, label in zip(bert_attrs[0], sst_attrs, labels):
        if i in short_samples_indices:
            continue
        res = evaluate_attr(bert_attr, sst_attr, label)
        for metric, result in res.items():
            evaluations[metric].append(result)
    i += 1

    for key in evaluations.keys():
        evaluations[key] = sum(evaluations[key]) / float(len(evaluations[key]))

    return evaluations


def main():
    output_csv_file = open(OUTPUT_FILE, 'w+', encoding='utf-8')

    method_file_dict = get_method_file_dict()
    generate_random_attrs(method_file_dict)
    phrase_sentiments = get_phrase_sentiments()
    sst_bert_tokens = get_tokens()
    sst_labels = get_sst_labels(sst_bert_tokens, phrase_sentiments)
    sst_attrs = get_sst_sentiments(sst_bert_tokens['sst_tokens'], phrase_sentiments)
    sst_attrs = scale_sst_attrs(sst_attrs)
    short_sample_indices = get_short_sample_indices(sst_bert_tokens)

    if ATTRS_ABS:
        sst_attrs = to_abs_values(sst_attrs)

    if LOWERCASE_SST:
        sst_bert_tokens['sst_tokens'] = lowercase_sst_tokens(sst_bert_tokens['sst_tokens'])

    output_csv_file.write('method;top1;top3;top5\n')
    # process attributions for each method
    for method, file in method_file_dict.items():
        if not ATTRS_ABS and method == 'relprop':
            continue
        
        attrs = load_json(ATTRS_DIR + '/' + file)

        evals = process_method(attrs, sst_attrs, short_sample_indices, sst_bert_tokens, sst_labels)
        output_csv_file.write(f'{method};' +
                              ';'.join('{:.3f}'.format(x) for x in evals.values()) +
                              '\n')

    output_csv_file.close()


if __name__ == '__main__':
    main()

import argparse
import json
import numpy as np
from utils.stemming import cz_stem


parser = argparse.ArgumentParser()
parser.add_argument('--attrs_dir', required=True, help='Output directory of the create_attributions_sst script')
parser.add_argument('--output_file', default='metrics.csv', help='File to write the results to')
parser.add_argument('--absolute', default=False, help='If True, absolute attributions, else Pos/Neg attributions')

args = parser.parse_args()

OUTPUT_FILE = args.output_file
np.random.seed(42)
ATTRS_DIR = args.attrs_dir
ATTRS_ABS = args.absolute

PMI_CUTOFF = 5
WORD_COUNT_CUTOFF = 75

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
    return load_json(ATTRS_DIR + '/bert_tokens.json')


def get_target_indices():
    return load_json(ATTRS_DIR + '/target_indices.json')


def to_abs_values(values):
    return rec_abs(values)


def get_classes():
    return load_json('datasets_ours/news/classes.json')


def lowercase_tokens(bert_tokens: list):
    lowercased = []
    for document in bert_tokens:
        tokens = []
        for token in document:
            tokens.append(token.lower())
        lowercased.append(tokens)
    return lowercased


def get_class_word_dict():
    classes = get_classes()
    with open('datasets_ours/news/PMI.csv', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    class_word_dict_temp = {}
    class_word_dict = {}
    for clss, idx in classes.items():
        class_word_dict[idx] = []
        class_word_dict_temp[idx] = []

    for i in range(len(lines)):
        if i == 0:
            continue
        split = lines[i].split(',')
        clss = split[1].lower()
        if clss not in classes.keys():
            continue
        pmi = float(split[6])
        if pmi < PMI_CUTOFF:
            continue
        idx = classes[clss]
        word = cz_stem(split[2].lower())                     # ignore phrases
        if word not in class_word_dict_temp[idx] and len(word.split()) == 1:
            class_word_dict_temp[idx].append(word)
            class_word_dict[idx].append((word, pmi))

    invalid_class_indices = []
    for clss, words in class_word_dict.items():
        if len(words) < WORD_COUNT_CUTOFF:
            invalid_class_indices.append(clss)
        print(f'{clss} - {len(words)}')

    return class_word_dict, invalid_class_indices


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
    attrs = load_json(ATTRS_DIR + "/" + attrs_file)
    random_attrs = []
    random_time = 0

    for sample in attrs[0]:
        sample_random = []
        for target_index in sample:
            shape = np.array(target_index).shape
            r = np.random.uniform(-0.5, 0.5, shape)
            sample_random.append(r.tolist())
        random_attrs.append(sample_random)

    res = [random_attrs, random_time]
    random_file = ATTRS_DIR + "/random.json"
    with open(random_file, 'w+', encoding='utf-8') as f:
        f.write(json.dumps(res))

    method_file_dict['random'] = 'random.json'


def preprocess_token_attrs(bert_tokens: list, attributions: list):
    processed_attributions = []
    processed_tokens = []
    # process each document
    for document, document_attrs in zip(bert_tokens, attributions[0]):
        # merge the tokens
        processed_document = []
        token = document[0]
        length = 1
        for i in range(1, len(document)):
            if '##' in document[i]:
                length += 1
                token += document[i][2:]
            else:
                processed_document.append(cz_stem(token))
                token = document[i]
                length = 1
        processed_tokens.append(processed_document)

        # merge the attributions for each class
        processed_document_attributions = []
        attrs_sum = document_attrs[0][0]
        length = 1
        for class_attrs in document_attrs:
            processed_class_attrs = []
            for i in range(1, len(class_attrs)):
                if '##' in document[i]:
                    length += 1
                    attrs_sum += class_attrs[i]
                else:
                    processed_class_attrs.append(attrs_sum / length)
                    attrs_sum = class_attrs[i]
                    length = 1
            processed_document_attributions.append(processed_class_attrs)
        processed_attributions.append(processed_document_attributions)

    return (processed_attributions, attributions[1]), processed_tokens


def merge_embedding_attrs(attributions: list):
    # go over documents
    for i in range(len(attributions[0])):
        # go over classes
        for j in range(len(attributions[0][i])):

            # if the shape has only one dimension, there is nothing to do
            if len(np.array(attributions[0][i][j]).shape) == 1:
                return attributions

            # go trough all token embeddings
            for k in range(len(attributions[0][i][j])):
                attr = np.array(attributions[0][i][j][k])
                attr = np.sum(attr, axis=0)
                attributions[0][i][j][k] = attr.tolist()

    return attributions


def get_word_attrs_dict(attrs, tokens):
    d = {}
    for token, attr in zip(tokens, attrs):
        if token in d.keys():
            d[token].append(attr)
        else:
            d[token] = [attr]

    for token in d.keys():
        d[token] = sum(d[token])

    return d


def get_keywords(tokens, words_pmi):
    tokens = [t.lower() for t in tokens]
    words = [w for w, pmi in words_pmi]
    keywords = [w for w in words if w in tokens]
    return keywords


def get_top_k_attrs(word_attr_dict: dict, k: int):
    sorted_words = [word for word, attr in sorted(word_attr_dict.items(), key=lambda item: item[1], reverse=True)]
    if len(sorted_words) <= k:
        return sorted_words, sorted_words
    else:
        return sorted_words[:k], sorted_words


def eval_top_k(attrs, tokens, words_pmi, k):
    word_attr_dict = get_word_attrs_dict(attrs, tokens)
    keywords = get_keywords(tokens, words_pmi)

    if len(keywords) == 0:
        return None

    top_k_words, all_words = get_top_k_attrs(word_attr_dict, k)
    top_k_keywords_intersect = set(keywords).intersection(set(top_k_words))
    return len(top_k_keywords_intersect) / len(keywords)


def evaluate_attr(bert_attrs: list, tokens: list, words_pmi: list):
    # evaluate the attributions for one sentence
    res = {
        'top5': eval_top_k(bert_attrs, tokens, words_pmi, 5),
        'top10': eval_top_k(bert_attrs, tokens, words_pmi, 10),
        'top15': eval_top_k(bert_attrs, tokens, words_pmi, 15),
    }

    return res


def process_method(bert_attrs: list, bert_tokens: list, class_words_dict: dict,  target_indices: list, invalid_class_indices):
    bert_attrs = merge_embedding_attrs(bert_attrs)
    bert_attrs, bert_tokens = preprocess_token_attrs(bert_tokens, bert_attrs)

    if ATTRS_ABS:
        bert_attrs = to_abs_values(bert_attrs)

    # evaluate the preprocessed attributions
    evaluations = {
        'top5': [],
        'top10': [],
        'top15': [],
    }

    for bert_attr, tokens, class_indices in zip(bert_attrs[0], bert_tokens, target_indices):
        for class_idx, attr in zip(class_indices, bert_attr):
            if class_idx in invalid_class_indices:
                continue
            res = evaluate_attr(attr, tokens, class_words_dict[class_idx])
            for metric, result in res.items():
                if result is None:
                    continue
                evaluations[metric].append(result)

    for metric in evaluations.keys():
        evaluations[metric] = sum(evaluations[metric]) / len(evaluations[metric])

    return evaluations


def main():
    output_csv_file = open(OUTPUT_FILE, 'w+', encoding='utf-8')

    method_file_dict = get_method_file_dict()
    generate_random_attrs(method_file_dict)
    bert_tokens = get_tokens()
    bert_tokens = lowercase_tokens(bert_tokens)
    classes_significant_words_dict, invalid_class_indices = get_class_word_dict()
    target_indices = get_target_indices()

    output_csv_file.write('method;top5;top10;top15\n')
    # process attributions for each method
    for method, file in method_file_dict.items():
        if not ATTRS_ABS and method == 'relprop':
            continue
        
        attrs = load_json(ATTRS_DIR + '/' + file)

        evals = process_method(attrs, bert_tokens, classes_significant_words_dict, target_indices, invalid_class_indices)
        output_csv_file.write(f'{method};' +
                              ';'.join('{:.3f}'.format(x) for x in evals.values()) +
                              '\n')
    output_csv_file.close()


if __name__ == '__main__':
    main()

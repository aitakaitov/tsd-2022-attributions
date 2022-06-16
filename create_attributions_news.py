import argparse

import torch
import json

from transformers import AutoTokenizer
from attribution_methods_custom import gradient_attributions, ig_attributions, sg_attributions
from models.bert_512 import BertSequenceClassifierNews, ElectraSequenceClassifierNews
import numpy as np
from BERT_explainability.modules.BERT.ExplanationGenerator import Generator

from time import perf_counter_ns
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', default='output_news_attrs', help='Attributions output directory')
parser.add_argument('--model_path', required=True, help='Trained model')
parser.add_argument('--part', required=False, default='all', help='Which split to compute - one of [g_sg20-50, sg100, sg200, rp_ig20-50, ig100, ig200, all]')

args = parser.parse_args()

OUTPUT_DIR = args.output_dir
MODEL_PATH = args.model_path
PART = args.part

try:
    os.mkdir(OUTPUT_DIR)
except OSError:
    pass

if 'small-e-czech' in MODEL_PATH:
    tokenizer = AutoTokenizer.from_pretrained('Seznam/small-e-czech')
    model = ElectraSequenceClassifierNews.from_pretrained(MODEL_PATH)
    embeddings = model.electra.base_model.embeddings.word_embeddings.weight.data
else:
    tokenizer = AutoTokenizer.from_pretrained('UWB-AIR/Czert-B-base-cased')
    model = BertSequenceClassifierNews.from_pretrained(MODEL_PATH)
    embeddings = model.bert.base_model.embeddings.word_embeddings.weight.data

ids = tokenizer.encode('[PAD]')
pad_token_index = ids[1]
cls_token_index = ids[0]
sep_token_index = ids[2]

embeddings = embeddings.to(device)
model = model.to(device)

model.eval()
relprop_explainer = Generator(model)

method_file_dict = {
    'grads': 'gradient_attrs_custom.json',
    'grads_x_inputs':  'gradients_x_inputs_attrs_custom.json',
    'ig_20':  'ig_20_attrs_custom.json',
    'ig_50':  'ig_50_attrs_custom.json',
    'ig_100':  'ig_100_attrs_custom.json',
    'ig_200':  'ig_200_attrs_custom.json',
    'sg_20':  'sg_20_attrs_custom.json',
    'sg_50':  'sg_50_attrs_custom.json',
    'sg_100':  'sg_100_attrs_custom.json',
    'sg_200':  'sg_200_attrs_custom_bk.json',
    'sg_20_x_inputs':  'sg_20_x_inputs_attrs_custom.json',
    'sg_50_x_inputs':  'sg_50_x_inputs_attrs_custom.json',
    'sg_100_x_inputs':  'sg_100_x_inputs_attrs_custom.json',
    'sg_200_x_inputs':  'sg_200_x_inputs_attrs_custom.json',
    'relprop':  'relprop_attrs.json'
}

#   -----------------------------------------------------------------------------------------------


def parse_csv_line(line: str):
    split = line.strip('\n').split('~')
    text = split[0]
    classes = split[1:]
    return text, classes


def get_data():
    with open('datasets_ours/news/classes.json', 'r', encoding='utf-8') as f:
        class_dict = json.loads(f.read())

    with open('datasets_ours/news/test.csv', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    samples = []
    labels = []
    for line in lines:
        text, classes = parse_csv_line(line)
        classes = [class_dict[clss] for clss in classes]
        samples.append(text)
        labels.append(classes)

    return samples, labels


def format_attrs(attrs, sentence):
    tokenized = tokenizer(sentence)

    if len(attrs.shape) == 2 and attrs.shape[0] == 1:
        attrs = torch.squeeze(attrs)

    attrs_list = attrs.tolist()
    return attrs_list[1:len(tokenized.data['input_ids']) - 1]  # leave out cls and sep


def prepare_embeds_and_att_mask(sentence):
    encoded = tokenizer(sentence, padding='max_length', max_length=512, truncation=True, return_tensors='pt')
    attention_mask = encoded.data['attention_mask'].to(device)
    input_ids = torch.squeeze(encoded.data['input_ids']).to(device)
    input_embeds = torch.unsqueeze(torch.index_select(embeddings, 0, input_ids), 0).requires_grad_(True).to(device)
    input_ids.to('cpu')

    return input_embeds, attention_mask


def prepare_input_ids_and_attention_mask(sentence):
    encoded = tokenizer(sentence, padding='max_length', max_length=512, truncation=True, return_tensors='pt')
    attention_mask = encoded.data['attention_mask'].to(device)
    input_ids = encoded.data['input_ids'].to(device)

    return input_ids, attention_mask


def create_neutral_baseline(sentence):
    length = len(tokenizer.tokenize(sentence))
    baseline_text = "".join(['[PAD]' for _ in range(length)])
    inputs_embeds, attention_mask = prepare_embeds_and_att_mask(baseline_text)
    return inputs_embeds


#   -----------------------------------------------------------------------------------------------

def create_gradient_attributions(sentences, target_indices_list):
    if PART != 'g_sg20-50' and PART != 'all':
        return

    file = open(OUTPUT_DIR + '/' + method_file_dict['grads'], 'w+', encoding='utf-8')
    file.write('[\n[\n')

    times = []
    for sentence, target_indices in zip(sentences, target_indices_list):
        input_embeds, attention_mask = prepare_embeds_and_att_mask(sentence)
        attrs_temp = []
        for target_idx in target_indices:
            start_time = perf_counter_ns()
            attr = gradient_attributions(input_embeds, attention_mask, target_idx, model)
            times.append(perf_counter_ns() - start_time)
            attr = torch.squeeze(attr)
            attrs_temp.append(format_attrs(attr, sentence))

        file.write(json.dumps(attrs_temp) + ',')

    file.seek(file.tell() - 1)
    file.write('\n]\n,' + json.dumps(float(np.average(times))) + '\n]\n')
    file.close()

    file = open(OUTPUT_DIR + '/' + method_file_dict['grads_x_inputs'], 'w+', encoding='utf-8')
    file.write('[\n[\n')

    times = []
    for sentence, target_indices in zip(sentences, target_indices_list):
        input_embeds, attention_mask = prepare_embeds_and_att_mask(sentence)
        attrs_temp = []
        for target_idx in target_indices:
            start_time = perf_counter_ns()
            attr = gradient_attributions(input_embeds, attention_mask, target_idx, model, True)
            times.append(perf_counter_ns() - start_time)
            attr = torch.squeeze(attr)
            attrs_temp.append(format_attrs(attr, sentence))

        file.write(json.dumps(attrs_temp) + ',')

    file.seek(file.tell() - 1)
    file.write('\n]\n,' + json.dumps(float(np.average(times))) + '\n]\n')
    file.close()


def _do_ig(sentences, target_indices_list, steps, file):
    file = open(OUTPUT_DIR + '/' + method_file_dict[file], 'w+', encoding='utf-8')
    file.write('[\n[\n')

    times = []
    for sentence, target_indices in zip(sentences, target_indices_list):
        input_embeds, attention_mask = prepare_embeds_and_att_mask(sentence)
        attrs_temp = []
        for target_idx in target_indices:

            start_time = perf_counter_ns()
            baseline = create_neutral_baseline(sentence)
            attr = ig_attributions(input_embeds, attention_mask, target_idx, baseline, model, steps)

            times.append(perf_counter_ns() - start_time)
            attr = torch.squeeze(attr)
            attrs_temp.append(format_attrs(attr, sentence))

        file.write(json.dumps(attrs_temp) + ',')

    file.seek(file.tell() - 1)
    file.write('\n]\n,' + json.dumps(float(np.average(times))) + '\n]\n')
    file.close()


def create_ig_attributions(sentences, target_indices):
    if PART == 'all':
        _do_ig(sentences, target_indices, 20, 'ig_20')
        _do_ig(sentences, target_indices, 50, 'ig_50')
        _do_ig(sentences, target_indices, 100, 'ig_100')
        _do_ig(sentences, target_indices, 200, 'ig_200')
    elif PART == 'rp_ig20-50':
        _do_ig(sentences, target_indices, 20, 'ig_20')
        _do_ig(sentences, target_indices, 50, 'ig_50')
    elif PART == 'ig100':
        _do_ig(sentences, target_indices, 100, 'ig_100')
    elif PART == 'ig200':
        _do_ig(sentences, target_indices, 200, 'ig_200')


def _do_sg(sentences, target_indices_list, samples, file):
    f = open(OUTPUT_DIR + '/' + method_file_dict[file], 'w+', encoding='utf-8')
    f.write('[\n[\n')

    f_x_inputs = open(OUTPUT_DIR + '/' + method_file_dict[file + '_x_inputs'], 'w+', encoding='utf-8')
    f_x_inputs.write('[\n[\n')

    times = []
    times_x_input = []
    for sentence, target_indices in zip(sentences, target_indices_list):
        inputs_embeds, attention_mask = prepare_embeds_and_att_mask(sentence)
        temp_attrs = []
        temp_attrs_x_inputs = []
        for target_idx in target_indices:
            start_time_1 = perf_counter_ns()
            attr = sg_attributions(inputs_embeds, attention_mask, target_idx, model, samples)
            end_time_1 = perf_counter_ns()
            start_time_2 = perf_counter_ns()
            attr_x_input = attr.to(device) * inputs_embeds
            end_time_2 = perf_counter_ns()
            attr_x_input = torch.squeeze(attr_x_input)
            attr = torch.squeeze(attr)
            temp_attrs.append(format_attrs(attr, sentence))
            temp_attrs_x_inputs.append(format_attrs(attr_x_input, sentence))

            times.append(end_time_1 - start_time_1)
            times_x_input.append((end_time_1 - start_time_1) + (end_time_2 - start_time_2))

        f.write(json.dumps(temp_attrs) + ',')
        f_x_inputs.write(json.dumps(temp_attrs_x_inputs) + ',')

    f.seek(f.tell() - 1)
    f.write('\n]\n,' + json.dumps(float(np.average(times))) + '\n]\n')
    f.close()

    f_x_inputs.seek(f_x_inputs.tell() - 1)
    f_x_inputs.write('\n]\n,' + json.dumps(float(np.average(times))) + '\n]\n')
    f_x_inputs.close()


def create_smoothgrad_attributions(sentences, target_indices):
    if PART == 'all':
        _do_sg(sentences, target_indices, 20, 'sg_20')
        _do_sg(sentences, target_indices, 50, 'sg_50')
        _do_sg(sentences, target_indices, 100, 'sg_100')
        _do_sg(sentences, target_indices, 200, 'sg_200')
    elif PART == 'g_sg20-50':
        _do_sg(sentences, target_indices, 20, 'sg_20')
        _do_sg(sentences, target_indices, 50, 'sg_50')
    elif PART == 'sg100':
        _do_sg(sentences, target_indices, 100, 'sg_100')
    elif PART == 'sg200':
        _do_sg(sentences, target_indices, 200, 'sg_200')


def create_relprop_attributions(sentences, target_indices_list):
    if PART != 'rp_ig20-50' and PART != 'all':
        return

    f = open(OUTPUT_DIR + '/' + method_file_dict['relprop'], 'w+', encoding='utf-8')
    f.write('[\n[\n')

    times = []
    for sentence, target_indices in zip(sentences, target_indices_list):
        input_ids, attention_mask = prepare_input_ids_and_attention_mask(sentence)
        inputs_embeds, _ = prepare_embeds_and_att_mask(sentence)
        temp_attrs = []
        for target_idx in target_indices:
            start_time = perf_counter_ns()
            res = relprop_explainer.generate_LRP(input_ids=input_ids, attention_mask=attention_mask, start_layer=0, index=target_idx)
            times.append(perf_counter_ns() - start_time)
            temp_attrs.append(format_attrs(res, sentence))

        f.write(json.dumps(temp_attrs) + ',')

    f.seek(f.tell() - 1)
    f.write('\n]\n,' + json.dumps(float(np.average(times))) + '\n]\n')
    f.close()

#   -----------------------------------------------------------------------------------------------


def main():
    documents, labels = get_data()

    bert_tokens = []
    target_indices = []
    valid_documents = []
    for document, label in zip(documents, labels):
        # first classify the sample
        input_embeds, attention_mask = prepare_embeds_and_att_mask(document)
        res = model(inputs_embeds=input_embeds, attention_mask=attention_mask, inputs_embeds_in_input_ids=False)
        res = list(torch.squeeze(res))
        target = []
        for i in range(len(res)):
            if res[i] >= 0.6 and i in label:
                target.append(i)
        if len(target) != 0:
            target_indices.append(target)
            bert_tokens.append(tokenizer.tokenize(document))
            valid_documents.append(document)

    # dump the tokens
    with open(OUTPUT_DIR + '/bert_tokens.json', 'w+', encoding='utf-8') as f:
        f.write(json.dumps(bert_tokens))

    with open(OUTPUT_DIR + '/target_indices.json', 'w+', encoding='utf-8') as f:
        f.write(json.dumps(target_indices))

    if 'czert' not in MODEL_PATH.lower():
        method_file_dict.pop('relprop')

    with open(OUTPUT_DIR + '/method_file_dict_custom.json', 'w+', encoding='utf-8') as f:
        f.write(json.dumps(method_file_dict))

    create_gradient_attributions(valid_documents, target_indices)
    create_smoothgrad_attributions(valid_documents, target_indices)
    create_ig_attributions(valid_documents, target_indices)

    if 'czert' in MODEL_PATH.lower():
        create_relprop_attributions(valid_documents, target_indices)


if __name__ == '__main__':
    main()

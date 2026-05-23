'''
Copyright 2025 trueWangSyutung
Open Academic Community License V1
'''
import argparse
import csv
import json
import os
import re

import numpy as np


DEFAULT_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'


def clean_text(value):
    if value is None:
        return ''
    if not isinstance(value, str):
        value = str(value)
    value = value.replace('\r', ' ').replace('\n', ' ').replace('\t', ' ')
    return re.sub(r'\s+', ' ', value).strip()


def format_languages(value):
    if not value:
        return ''
    if isinstance(value, dict):
        ordered = sorted(value.items(), key=lambda item: item[1], reverse=True)
        return ', '.join(clean_text(language) for language, _ in ordered if clean_text(language))
    return clean_text(value)


def format_bool_flag(name, value):
    if value is None:
        return ''
    return name if bool(value) else 'no ' + name


def format_github_text(github):
    if not isinstance(github, dict):
        return ''

    parts = []
    full_name = clean_text(github.get('full_name'))
    language = clean_text(github.get('language'))
    languages = format_languages(github.get('languages'))
    license_name = clean_text(github.get('license'))

    if full_name:
        parts.append('GitHub repository: {}.'.format(full_name))
    if language:
        parts.append('Primary language: {}.'.format(language))
    if languages:
        parts.append('Languages: {}.'.format(languages))
    if license_name:
        parts.append('License: {}.'.format(license_name))

    signals = []
    for key, label in [
        ('stargazers_count', 'stars'),
        ('forks_count', 'forks'),
        ('contributors_count', 'contributors'),
        ('open_issues_count', 'open issues'),
    ]:
        value = github.get(key)
        if value is not None:
            signals.append('{} {}'.format(value, label))
    if signals:
        parts.append('Repository signals: {}.'.format(', '.join(signals)))

    status = []
    if github.get('archived') is not None:
        status.append('archived repository' if github.get('archived') else 'active repository')
    for key, label in [
        ('has_docker', 'has Docker'),
        ('has_readme', 'has README'),
        ('has_requirements', 'has requirements file'),
    ]:
        flag = format_bool_flag(label, github.get(key))
        if flag:
            status.append(flag)
    last_commit = clean_text(github.get('last_commit'))
    if last_commit:
        status.append('last commit {}'.format(last_commit))
    if status:
        parts.append('Repository status: {}.'.format(', '.join(status)))

    return ' '.join(parts)


def build_attribute_text(server, attribute_set):
    fields = []
    for label, key in [
        ('MCP server name', 'name'),
        ('Title', 'title'),
        ('Category', 'category'),
        ('Description', 'description'),
        ('Tags', 'tags'),
        ('Author', 'author_name'),
        ('URL', 'url'),
    ]:
        value = clean_text(server.get(key))
        if value:
            fields.append('{}: {}.'.format(label, value))

    github_text = ''
    if attribute_set == 'AB':
        github_text = format_github_text(server.get('github'))
        if github_text:
            fields.append(github_text)

    return ' '.join(fields), github_text


def load_id_mapping(path):
    mapping = {}
    with open(path, newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        if 'id' not in reader.fieldnames or 'iid' not in reader.fieldnames:
            raise ValueError('Mapping file must contain id and iid columns.')
        for row in reader:
            mapping[int(row['id'])] = int(row['iid'])
    return mapping


def load_servers(path):
    with open(path, encoding='utf-8') as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError('Server JSON must be a list of server objects.')
    return data


def write_attribute_csv(rows, path):
    fieldnames = [
        'iid',
        'id',
        'attribute_set',
        'name',
        'title',
        'category',
        'tags',
        'description',
        'author_name',
        'url',
        'github_text',
        'attribute_text',
    ]
    with open(path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def encode_texts(texts, model_name, batch_size):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return embeddings.astype('float32')


def prepare_rows(servers, id_to_iid, attribute_set):
    rows = []
    missing_mapping = []
    duplicate_json_ids = []
    seen_raw_ids = set()
    seen_iids = set()

    for server in servers:
        raw_id = server.get('id')
        if raw_id is None:
            continue
        raw_id = int(raw_id)
        if raw_id in seen_raw_ids:
            duplicate_json_ids.append(raw_id)
            continue
        seen_raw_ids.add(raw_id)
        if raw_id not in id_to_iid:
            missing_mapping.append(raw_id)
            continue

        iid = id_to_iid[raw_id]
        attribute_text, github_text = build_attribute_text(server, attribute_set)
        rows.append({
            'iid': iid,
            'id': raw_id,
            'attribute_set': attribute_set,
            'name': clean_text(server.get('name')),
            'title': clean_text(server.get('title')),
            'category': clean_text(server.get('category')),
            'tags': clean_text(server.get('tags')),
            'description': clean_text(server.get('description')),
            'author_name': clean_text(server.get('author_name')),
            'url': clean_text(server.get('url')),
            'github_text': github_text,
            'attribute_text': attribute_text,
        })
        seen_iids.add(iid)

    rows = sorted(rows, key=lambda row: row['iid'])
    expected_iids = set(id_to_iid.values())
    missing_iids = sorted(expected_iids - seen_iids)
    return rows, sorted(missing_mapping), missing_iids, sorted(duplicate_json_ids)


def save_feature_matrix(rows, feature_path, model_name, batch_size):
    max_iid = max(row['iid'] for row in rows)
    texts = [row['attribute_text'] for row in rows]
    row_iids = [row['iid'] for row in rows]
    encoded = encode_texts(texts, model_name, batch_size)

    features = np.zeros((max_iid + 1, encoded.shape[1]), dtype='float32')
    for row_index, iid in enumerate(row_iids):
        features[iid] = encoded[row_index]
    np.save(feature_path, features)
    return features.shape


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--servers_json', type=str, default='D:/EEE/pycharm/MCPCorpus/Website/filtered_servers.json')
    parser.add_argument('--id_mapping', type=str, default='D:/EEE/pycharm/MCPCorpus/Website/filtered_servers_id_mapping.csv')
    parser.add_argument('--output_dir', type=str, default='data/mcp_item_attributes')
    parser.add_argument('--model_name', type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--attribute_sets', nargs='+', default=['A', 'AB'], choices=['A', 'AB'])
    parser.add_argument('--text_only', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    servers = load_servers(args.servers_json)
    id_to_iid = load_id_mapping(args.id_mapping)

    summary = {
        'servers_json': args.servers_json,
        'id_mapping': args.id_mapping,
        'output_dir': args.output_dir,
        'model_name': args.model_name,
        'attribute_sets': args.attribute_sets,
        'num_servers_in_json': len(servers),
        'num_mapped_items': len(id_to_iid),
        'outputs': {},
    }

    for attribute_set in args.attribute_sets:
        rows, missing_mapping, missing_iids, duplicate_json_ids = prepare_rows(servers, id_to_iid, attribute_set)
        suffix = attribute_set
        csv_path = os.path.join(args.output_dir, 'item_attribute_text_{}.csv'.format(suffix))
        feature_path = os.path.join(args.output_dir, 'item_text_features_{}.npy'.format(suffix))

        write_attribute_csv(rows, csv_path)
        feature_shape = None
        if not args.text_only:
            feature_shape = save_feature_matrix(rows, feature_path, args.model_name, args.batch_size)

        summary['outputs'][attribute_set] = {
            'attribute_csv': csv_path,
            'feature_npy': None if args.text_only else feature_path,
            'feature_shape': feature_shape,
            'rows': len(rows),
            'missing_json_ids_without_mapping': missing_mapping[:20],
            'num_missing_json_ids_without_mapping': len(missing_mapping),
            'missing_mapped_iids_without_json': missing_iids[:20],
            'num_missing_mapped_iids_without_json': len(missing_iids),
            'duplicate_json_ids_skipped': duplicate_json_ids[:20],
            'num_duplicate_json_ids_skipped': len(duplicate_json_ids),
        }

    meta_path = os.path.join(args.output_dir, 'item_feature_meta.json')
    with open(meta_path, 'w', encoding='utf-8') as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

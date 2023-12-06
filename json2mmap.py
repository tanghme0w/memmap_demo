import json
import numpy as np


def feat2mmap(feat_file, mmap_feat_file, mmap_index_file, max_item_id=-1, max_embedding_size=-1, flush_interval=1000):
    # count total number of items
    num_items = sum(1 for line in open(feat_file, 'r'))

    # find the largest item_id (if not assigned in advance)
    if max_item_id == -1:
        max_item_id = max(json.loads(line)['text_id'] for line in open(feat_file, 'r'))

    # find the largest embedding size (if not assigned in advance)
    if max_embedding_size == -1:
        max_embedding_size = max(len(json.loads(line)['text_feature']) for line in open(feat_file, 'r'))

    # create feature & index mmap
    feat_mmap = np.memmap(mmap_feat_file, dtype=int, mode='w+', shape=(num_items, max_embedding_size))
    index_mmap = np.memmap(mmap_index_file, dtype=int, mode='w+', shape=(max_item_id + 1,))

    # read feature file and construct mmap line by line
    with open(feat_file, 'r') as ff:
        for i, line in enumerate(ff):
            feat_dict = json.loads(line)
            text_id = feat_dict['text_id']
            text_feat = feat_dict['text_feature']
            feat_mmap[i, :] = text_feat
            index_mmap[text_id] = i
            if i % flush_interval == 0:
                print(f'flushing data on line #{i}')
                feat_mmap.flush()
                index_mmap.flush()


feat2mmap('feature.jsonl', 'mmap_feat', 'mmap_index')

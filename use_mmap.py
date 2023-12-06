import numpy as np
import random
import json

feat_file = 'feature.jsonl'
mmap_feat_file = 'mmap_feat'
mmap_index_file = 'mmap_index'

# count total number of items
num_items = sum(1 for line in open(feat_file, 'r'))

# find the largest item_id (if not assigned in advance)
max_item_id = max(json.loads(line)['text_id'] for line in open(feat_file, 'r'))

# find the largest embedding size (if not assigned in advance)
max_embedding_size = max(len(json.loads(line)['text_feature']) for line in open(feat_file, 'r'))

# create feature & index mmap
feat_mmap = np.memmap(mmap_feat_file, dtype=int, mode='r', shape=(num_items, max_embedding_size))
index_mmap = np.memmap(mmap_index_file, dtype=int, mode='r', shape=(max_item_id + 1,))

print(feat_mmap.shape, index_mmap.shape)


def emb_lookup(index):
    mmap_index = index_mmap[index]
    print(mmap_index)
    return feat_mmap[mmap_index]


random_indices = list(range(100))
random.shuffle(random_indices)
print(emb_lookup(1))

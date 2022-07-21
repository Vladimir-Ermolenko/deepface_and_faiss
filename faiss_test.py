import random
import time

import faiss
import pickle
import numpy as np


with open('C:\\Users\\neuro_srv2\\Downloads\\photos\\representations_arcface.pkl', 'rb') as f:
    data = pickle.load(f)

ids = []

vectors = []
vector_num = random.randint(1, 50000)
vector = []
image_name = []
counter = 0
for inner_lst in data:
    # inner_lst[0] = inner_lst[0].replace('/', '\\')
    counter += 1
    vectors.append(inner_lst[1])
    ids.append(inner_lst[0][inner_lst[0].rfind('/') + 1:inner_lst[0].rfind('.')])
    if counter == vector_num:
        vector.append(inner_lst[1])
        image_name.append(inner_lst[0][inner_lst[0].rfind('/') + 1:])

ids = np.array(ids, dtype=np.int64)
vectors = np.array(vectors, dtype=np.float32)
vector = np.array(vector, dtype=np.float32)

index = faiss.index_factory(512, 'IVF1000,Flat', faiss.METRIC_INNER_PRODUCT)
# index = faiss.IndexIDMap(index)
index.train(vectors)
index.add_with_ids(vectors, ids)

# start = time.time()
# faiss.write_index(index, 'C:\\Users\\neuro_srv2\\PycharmProjects\\deepface_and_faiss\\index.index')
# index = faiss.read_index('C:\\Users\\neuro_srv2\\PycharmProjects\\deepface_and_faiss\\index.index')
# print(time.time() - start)

distances, indexes = index.search(vector, 5)
# faiss.vector_to_array(index.id_map)
print(indexes)
print(distances)

print(image_name)

import glob
import random

import cv2
import faiss
import numpy as np

from arcface_embedding.arcface_embedding import ArcfaceEmbedding
from retina_face_detector.face_detector import RetinaFaceDetector

N_LIST = 500
N_VECTOR = N_LIST * 50
DIM = 512

if __name__ == '__main__':
    list_img = glob.glob('/home/nguyen/DATA/Dataset/CASIA-maxpy-clean/*/*.jpg')
    random.shuffle(list_img)
    n_img = len(list_img)
    print('There are', n_img, 'images')
    face_detector = RetinaFaceDetector(gpuid=0)
    face_emb = ArcfaceEmbedding()
    vector_count = 0
    list_vector = []
    for fn in list_img:
        print(fn)
        img = cv2.imread(fn)
        if img.shape[0] == 0 or img.shape[1] == 0:
            print('Cannot read', fn)
            continue
        detection_result = face_detector.detect(img)
        if detection_result is not None:
            recs, points = detection_result
            for rec, point in zip(recs, points):
                emb = face_emb.get_emb(img, rec, point)
                list_vector.append(emb)
                vector_count += 1
        print('vector_count:', vector_count)
        if vector_count >= N_VECTOR:
            break

    list_vector = np.array(list_vector)

    quantizer = faiss.IndexFlatL2(DIM)
    index = faiss.IndexIVFFlat(quantizer, DIM, N_LIST)
    print('Traning index')
    index.train(list_vector)
    print(index.is_trained)
    faiss.write_index(index, 'index_file.index')
    print('Done')

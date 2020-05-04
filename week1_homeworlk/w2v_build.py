# coding=utf-8
from gensim.models import Word2Vec
import os
import pickle
from gensim.models.word2vec import LineSentence
from gensim.models.keyedvectors import KeyedVectors

# 序列化保存数据
def dump_pkl(vocab, pkl_path, over_write=True):
    if pkl_path and os.path.exists(pkl_path) and not over_write:
        return
    if pkl_path:
        with open(pkl_path, 'wb') as fp:
            pickle.dump(vocab, fp, protocol=pickle.HIGHEST_PROTOCOL)
        print("save {} ok".format(pkl_path))
# 读取分词后的数据，放在一个list中
def read_lines(path):
    lines = list()
    fp = open(path, 'r', encoding='utf-8')
    for line in fp.readlines():
        line = line.strip()
        lines.append(line)
    return lines

def save_sentence(train_x_seg_path, train_y_seg_path, test_seg_path, sentence_path):
    f0 = open(sentence_path, 'a', encoding='utf-8')
    f1 = open(train_x_seg_path, 'r', encoding='utf-8')
    f2 = open(train_y_seg_path, 'r', encoding='utf-8')
    f3 = open(test_seg_path, 'r', encoding='utf-8')
    for f_1 in f1:
        f0.write(f_1)
    for f_2 in f2:
        f0.write(f_2)
    for f_3 in f3:
        f0.write(f_3)
    f0.close()
    f1.close()
    f2.close()
    f3.close()

# 训练词向量
def build(sentences_path, w2v_bin_path="w2v.bin", out_w2v_path=None):
    w2v = Word2Vec(sentences=LineSentence(sentences_path), size=256, window=5, min_count=20, iter=10, sg=1)
    w2v.wv.save_word2vec_format(w2v_bin_path, binary=True)
    print("save %s ok." % w2v_bin_path)
    sim = w2v.wv.similarity('技师', '车主')
    print('技师 vs 车主 similarity score:', sim)
    # load model
    model = KeyedVectors.load_word2vec_format(w2v_bin_path, binary=True)
    print(model.vocab)
    word_dict = {}
    for word in model.vocab:
        # print(word)
        word_dict[word] = model[word]
    dump_pkl(word_dict, out_w2v_path, over_write=True)

if __name__ == '__main__':
    save_sentence('./train_data_y.txt', './train_data_x.txt', './test_data_x.txt', './sentences.txt')
    build(sentences_path='./sentences.txt', out_w2v_path='./word2vc.txt')
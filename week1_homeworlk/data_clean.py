# coding=utf-8
import pandas
import jieba
'''
清洗数据
1.删除Report缺少数据所在行
2.分词之后，删除噪音词和停止词
3.保证训练集输入x,和输出y的数据行数相同，将中性词填补缺少的输出y
'''
# 噪音词
REMOVE_WORDS = ["[", "语音", "]", "|", "图片"]

# 读取停止词
def read_stopwords(path):
    lines = set()
    with open(path, mode='r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            lines.add(line)
    return lines

# 去除噪音词
def remove_worlds(worlds_list):
    return [world for world in worlds_list if world not in REMOVE_WORDS]

# 删除缺少Report数据行，返回训练集输入的x和输出y
def parse_data(train_data_path, test_data_path):
    # 1.读取数据
    tr_data = pandas.read_csv(train_data_path)
    # 2.删除report为空的数据行
    tr_data.dropna(inplace=True, subset=["Report"])
    # 3.填充为空的特征值
    tr_data.fillna(value='', inplace=True)
    # 4.将Question和Dialogue拼接
    tr_x = tr_data.Question.str.cat(tr_data.Dialogue)
    tr_y = tr_data["Report"]

    te_data = pandas.read_csv(test_data_path)
    te_data.fillna(value='', inplace=True)
    te_x = te_data.Question.str.cat(te_data.Dialogue)

    return tr_x, tr_y, te_x


def save_data(train_data_x, train_data_y, test_data_x, train_x_path, train_y_path, test_x_path, stop_word_path):
    # 读取停止词
    stop_worlds = read_stopwords(stop_word_path)

    # 1.保存训练集输入x的分词数据
    fp = open(train_x_path, 'w', encoding="utf-8")
    # 遍历每行训练集输入x
    count_1 = 0
    for data in train_data_x:
        # 对每行数据进行分词
        seword = jieba.lcut(data.strip())
        # 去除噪音词
        seword = remove_worlds(seword)
        # print(seword)
        # 去除停止词
        seword_list = [word for word in seword if word not in stop_worlds]
        if seword_list:
            seword = ' '.join(seword_list)
            fp.write(seword + '\n')
            count_1 += 1
    fp.close()
    print('x一共有{}条数据'.format(count_1))

    # 2.保存训练集输入x的分词数据
    fp = open(train_y_path, 'w', encoding="utf-8")
    count_2 = 0
    # 遍历每行训练集输入x
    for data in train_data_y:
        # 对每行数据进行分词
        seword = jieba.lcut(data.strip())
        # 去除噪音词
        seword = remove_worlds(seword)
        # print(seword)
        # 去除停止词
        seword_list = [word for word in seword if word not in stop_worlds]
        if seword_list:
            seword = ' '.join(seword_list)
            fp.write(seword + '\n')
            count_2 += 1
        else:
            seword = '随时 联系'
            fp.write(seword + '\n')
            count_2 += 1
    fp.close()
    # 实际运行如果不做处理，训练集输出y比输入x少5条
    print('y一共有{}条数据'.format(count_2))
    # 3.保存测试集输入x的分词数据
    fp = open(test_x_path, 'w', encoding="utf-8")
    count_3 = 0
    # 遍历每行训练集输入x
    for data in test_data_x:
        # 对每行数据进行分词
        seword = jieba.lcut(data.strip())
        # 去除噪音词
        seword = remove_worlds(seword)
        # print(seword)
        # 去除停止词
        seword_list = [word for word in seword if word not in stop_worlds]
        if seword_list:
            seword = ' '.join(seword_list)
            fp.write(seword + '\n')
            count_3 += 1
    fp.close()
    print('x2一共有{}条数据'.format(count_3))
if __name__ == '__main__':
    train_data_x, train_data_y, test_data_x = parse_data('../代码20200419/AutoMaster_TrainSet.csv',
                                                         '../代码20200419/AutoMaster_TestSet.csv')
    save_data(train_data_x, train_data_y, test_data_x, './train_data_x.txt', './train_data_y.txt','./test_data_x.txt', './stop_words.txt')
    # stopwords = read_stopwords('./stop_words.txt')
    # print(stopwords)
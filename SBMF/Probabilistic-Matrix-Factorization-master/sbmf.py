# -*- coding: utf-8 -*-
import numpy as np


class PMF(object):
    def __init__(self, num_feat=10, epsilon=1, _lambda=0.1, momentum=0.8, maxepoch=20, num_batches=10, batch_size=1000):
        # 样本中的所有样本数据被计算一次就叫做一个Epoch
        # 把样本数据分为若干批，分批来计算损失函数和更新参数，这样方向比较稳定，计算开销也相对较小。
        # BatchSize就是每一批的样本数量。
        self.num_feat = num_feat  # Number of latent features,
        self.epsilon = epsilon  # learning rate,
        self._lambda = _lambda  # L2 regularization,
        self.momentum = momentum  # momentum of the gradient,
        self.maxepoch = maxepoch  # Number of epoch before stop,
        self.num_batches = num_batches  # Number of batches in each epoch (for SGD optimization),
        self.batch_size = batch_size  # Number of training samples used in each batches (for SGD optimization)

        self.w_Item = None  # Item feature vectors
        self.w_User = None  # User feature vectors

        self.rmse_train = []
        self.rmse_test = []

    # ***Fit the model with train_tuple and evaluate RMSE on both train and test data.  ***********#
    # ***************** train_vec=TrainData, test_vec=TestData*************#
    def fit(self, train_vec, test_vec):
        # mean subtraction
        self.mean_inv = np.mean(train_vec[:, 2])  # 评分平均值
        self.mean_invs = np.mean(train_vec[:, 3])  # 评论平均值

        pairs_train = train_vec.shape[0]  # traindata 中条目数
        pairs_test = test_vec.shape[0]  # testdata中条目数

        # 1-p-i, 2-m-c
        num_user = int(max(np.amax(train_vec[:, 0]), np.amax(test_vec[:, 0]))) + 1  # 第0列，user总数
        num_item = int(max(np.amax(train_vec[:, 1]), np.amax(test_vec[:, 1]))) + 1  # 第1列，movie总数

        incremental = False  # 增量
        if ((not incremental) or (self.w_Item is None)):
            # initialize
            self.epoch = 0
            self.w_Item = 0.1 * np.random.randn(num_item, self.num_feat)  # numpy.random.randn 电影 M x D 正态分布矩阵
            self.w_User = 0.1 * np.random.randn(num_user, self.num_feat)  # numpy.random.randn 用户 N x D 正态分布矩阵

            self.w_Item_inc = np.zeros((num_item, self.num_feat))  # 创建电影 M x D 0矩阵
            self.w_User_inc = np.zeros((num_user, self.num_feat))  # 创建用户 N x D 0矩阵

        while self.epoch < self.maxepoch:  # 检查迭代次数
            self.epoch += 1

            # eg:在训练集中抽出一行 5,10,5，对应在 U 的第五行和 V 的第六行对应相乘求和
            # 打乱训练集，每一批取训练集1000行，同时取对应的Uid和Vid
            # 取出U和V中Uid和Vid对应的行进行相乘相加
            # Shuffle training truples
            shuffled_order = np.arange(train_vec.shape[0])# 根据训练集行数创建等差array
            np.random.shuffle(shuffled_order)  # 用于将一个列表中的元素打乱

            # Batch update
            for batch in range(self.num_batches):  # 每次迭代要使用的数据量
                # print "epoch %d batch %d" % (self.epoch, batch+1)

                test = np.arange(self.batch_size * batch, self.batch_size * (batch + 1))
                # test为本轮序号，如1000-2000,2000-3000
                batch_idx = np.mod(test, shuffled_order.shape[0])  # 本次迭代要使用的索引下标
                # 本轮训练个数对训练个数取模，如 (1000，2000) 对 79902 取模，等于1-1000,1000-2000
                # [shuffled_order[batch_idx], 0] 在打乱的79902行中取本轮用的1000行的第0列和第1列
                # 因此随机取得的用户ID和物品ID，1000个，肯定会取到重复的
                batch_UserID = np.array(train_vec[shuffled_order[batch_idx], 0], dtype='int32')
                batch_ItemID = np.array(train_vec[shuffled_order[batch_idx], 1], dtype='int32')
                # w_User 944 * 10
                x = np.multiply(self.w_User[batch_UserID, :],
                                              self.w_Item[batch_ItemID, :])
                # np.multiply对应元素相乘，要求两个矩阵的的形状shape相同，因此乘出来的矩阵形状也和原矩阵相同，1000 * 10
                pred_out = np.sum(np.multiply(self.w_User[batch_UserID, :],
                                              self.w_Item[batch_ItemID, :]),
                                  axis=1)  #axis=1对矩阵每一列上的元素进行求和，最后生成 1000 * 1 的向量
                # 误差函数,为什么要加self.mean_inv
                # train_vec[shuffled_order[batch_idx], 2] 从训练集中取出来的1000行的rating值
                rawErr = pred_out - train_vec[shuffled_order[batch_idx], 2] + self.mean_inv

                weight = train_vec[shuffled_order[batch_idx], 4]
                weights = 1 - train_vec[shuffled_order[batch_idx], 4]

                rawErr = pred_out - train_vec[shuffled_order[batch_idx], 2] + self.mean_inv
                rawErrS = pred_out - train_vec[shuffled_order[batch_idx], 3] + self.mean_invs

                wrawErr = np.multiply(rawErr, weight)
                wrawErrS = np.multiply(rawErrS, weights)

                Ix_Item = 2 * np.multiply(wrawErr[:, np.newaxis], self.w_User[batch_UserID, :]) + 2 * np.multiply(
                    wrawErrS[:, np.newaxis], self.w_User[batch_UserID, :]) + self._lambda * self.w_Item[batch_ItemID, :]
                Ix_User = 2 * np.multiply(wrawErr[:, np.newaxis], self.w_Item[batch_ItemID, :]) + 2 * np.multiply(
                    wrawErrS[:, np.newaxis], self.w_Item[batch_ItemID, :]) + self._lambda * self.w_User[batch_UserID, :]

                dw_Item = np.zeros((num_item, self.num_feat))
                dw_User = np.zeros((num_user, self.num_feat))

                # loop to aggreate the gradients of the same element
                for i in range(self.batch_size):
                    dw_Item[batch_ItemID[i], :] += Ix_Item[i, :]
                    dw_User[batch_UserID[i], :] += Ix_User[i, :]

                # Update with momentum
                self.w_Item_inc = self.momentum * self.w_Item_inc + self.epsilon * dw_Item / self.batch_size
                self.w_User_inc = self.momentum * self.w_User_inc + self.epsilon * dw_User / self.batch_size

                self.w_Item = self.w_Item - self.w_Item_inc
                self.w_User = self.w_User - self.w_User_inc

                # Compute Objective Function after
                # 每一轮迭代最后一批计算优化函数
                if batch == self.num_batches - 1:
                    pred_out = np.sum(np.multiply(self.w_User[np.array(train_vec[:, 0], dtype='int32'), :],
                                                  self.w_Item[np.array(train_vec[:, 1], dtype='int32'), :]),
                                      axis=1)  # mean_inv subtracted
                    rawErr = pred_out - train_vec[:, 2] + self.mean_inv
                    # 优化函数
                    # np.linalg.norm对矩阵求范数
                    obj = np.linalg.norm(rawErr) ** 2 \
                          + 0.5 * self._lambda * (np.linalg.norm(self.w_User) ** 2 + np.linalg.norm(self.w_Item) ** 2)

                    self.rmse_train.append(np.sqrt(obj / pairs_train))

                # Compute validation error
                if batch == self.num_batches - 1:
                    pred_out = np.sum(np.multiply(self.w_User[np.array(test_vec[:, 0], dtype='int32'), :],
                                                  self.w_Item[np.array(test_vec[:, 1], dtype='int32'), :]),
                                      axis=1)  # mean_inv subtracted
                    rawErr = pred_out - test_vec[:, 2] + self.mean_inv
                    # 训练集计算优化函数的误差，测试集计算预测分数之间的绝对误差
                    # self.rmse_test.append(np.sqrt(obj / pairs_test))
                    self.rmse_test.append(np.linalg.norm(rawErr) / np.sqrt(pairs_test))

                    # Print info
                    if batch == self.num_batches - 1:
                        print('Training RMSE: %f, Test RMSE %f' % (self.rmse_train[-1], self.rmse_test[-1]))

    def predict(self, invID):
        return np.dot(self.w_Item, self.w_User[int(invID), :]) + self.mean_inv  # numpy.dot 点乘

    # ****************Set parameters by providing a parameter dictionary.  ***********#
    def set_params(self, parameters):
        if isinstance(parameters, dict):
            self.num_feat = parameters.get("num_feat", 10)
            self.epsilon = parameters.get("epsilon", 1)
            self._lambda = parameters.get("_lambda", 0.1)
            self.momentum = parameters.get("momentum", 0.8)
            self.maxepoch = parameters.get("maxepoch", 20)
            self.num_batches = parameters.get("num_batches", 10)
            self.batch_size = parameters.get("batch_size", 1000)

    def topK(self, test_vec, k=10):
        inv_lst = np.unique(test_vec[:, 0])
        pred = {}
        for inv in inv_lst:
            if pred.get(inv, None) is None:
                pred[inv] = np.argsort(self.predict(inv))[-k:]  # numpy.argsort索引排序

        intersection_cnt = {}
        for i in range(test_vec.shape[0]):
            if test_vec[i, 1] in pred[test_vec[i, 0]]:
                intersection_cnt[test_vec[i, 0]] = intersection_cnt.get(test_vec[i, 0], 0) + 1
        invPairs_cnt = np.bincount(np.array(test_vec[:, 0], dtype='int32'))

        precision_acc = 0.0
        recall_acc = 0.0
        for inv in inv_lst:
            precision_acc += intersection_cnt.get(inv, 0) / float(k)
            recall_acc += intersection_cnt.get(inv, 0) / float(invPairs_cnt[int(inv)])

        return precision_acc / len(inv_lst), recall_acc / len(inv_lst)

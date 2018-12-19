import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D     # 画 3D 图的功能
import numpy as np
from LoadData import load_rating_data, spilt_rating_dat
from sklearn.model_selection import train_test_split
# from ProbabilisticMatrixFactorization import PMF
from sbmf import PMF

if __name__ == "__main__":
    file_path = "data/finalpatio.json"
    pmf = PMF()
    pmf.set_params({"num_feat": 10, "epsilon":0.05, "_lambda": 0.01, "momentum": 0.8, "maxepoch":50, "num_batches": 100,
                    "batch_size": 1000})
    print("load_rating_data")
    ratings = load_rating_data(file_path)
    print(len(np.unique(ratings[:, 0])), len(np.unique(ratings[:, 1])), pmf.num_feat)
    train, test = spilt_rating_dat(ratings)
    pmf.fit(train, test)

    # Check performance by plotting train and test errors
    plt.plot(range(pmf.maxepoch), pmf.rmse_train, marker='o', label='Training Data')
    plt.plot(range(pmf.maxepoch), pmf.rmse_test, marker='v', label='Test Data')
    plt.title('The MovieLens Dataset Learning Curve')
    plt.xlabel('Number of Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid()
    plt.show()
    print("precision_acc,recall_acc:" + str(pmf.topK(test)))




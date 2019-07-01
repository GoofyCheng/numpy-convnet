from multiprocessing import Pool
from model import *
from dataset_load import *
from solve import *


# 指定一个模型进行训练
def multitrain(data):
    model = SimpleCovnNet(reg=1e-3, hidden_dim=500, params_path="conv_params.txt")
    solve = Fit(model, data,
                lr_decay=0.95, print_every=100, num_epochs=4, batch_size=32,
                update_rule="adam",
                optim_config={"learning_rate": 1e-3})
    solve.train()
    return [solve.best_val_acc, solve.model.param_config, model]


if __name__ == "__main__":
    # 构造进程池
    p = Pool(3)
    process = []
    data = get_cifar_data(num_training=16000, num_test=2000)
    for i in range(4):
        process.append(p.apply_async(multitrain, (data,)))
    p.close()
    p.join()
    # print("模型处理完")
    # 挑选进程中表现效果最好的模型，并记录参数存储在本地文件中，方便后续继续读取该参数
    best_model_inf = process[0].get()
    for i in range(1, 3):
        a = process[i].get()
        if a[0] > best_model_inf[0]:
            best_model_inf = a
    best_model = best_model_inf[2]
    best_param = best_model_inf[1]
    # 存储参数至本地，可接着训练
    with open(best_model.params_path, "w") as f:
        f.write(str(best_param))
    y_test_predict = np.argmax(best_model.loss(data["x_test"]), axis=1)
    y_val_predict = np.argmax(best_model.loss(data["x_val"]), axis=1)
    y_test_acc = np.mean(y_test_predict == data["y_test"])
    y_val_acc = np.mean(y_val_predict == data["y_val"])
    print("test accuracy: %f" % y_test_acc)
    print("validation accuracy: %f" % y_val_acc)
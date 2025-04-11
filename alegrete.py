import numpy as np


def compute_mse(b, w, data):
    """
    Calcula o erro quadratico medio
    :param b: float - bias (intercepto da reta)
    :param w: float - peso (inclinacao da reta)
    :param data: np.array - matriz com o conjunto de dados, x na coluna 0 e y na coluna 1
    :return: float - o erro quadratico medio
    """
    x = data[:, 0]
    y = data[:, 1]

    y_pred = w*x +b

    mse = np.mean((y - y_pred)**2)

    return mse


def step_gradient(b, w, data, alpha):
    """
    Executa uma atualização por descida do gradiente  e retorna os valores atualizados de b e w.
    :param b: float - bias (intercepto da reta)
    :param w: float - peso (inclinacao da reta)
    :param data: np.array - matriz com o conjunto de dados, x na coluna 0 e y na coluna 1
    :param alpha: float - taxa de aprendizado (a.k.a. tamanho do passo)
    :return: float,float - os novos valores de b e w, respectivamente
    """
    x = data[:, 0]
    y = data[:, 1]

    N = len(data)

    y_pred = w*x + b
    errors = y_pred - y

    b_gradient = (2/N) * np.sum(errors)
    w_gradient = (2/N) * np.dot(errors, x)

    b = b - alpha * b_gradient
    w = w - alpha * w_gradient

    return b, w 


def fit(data, b, w, alpha, num_iterations):
    """
    Para cada época/iteração, executa uma atualização por descida de
    gradiente e registra os valores atualizados de b e w.
    Ao final, retorna duas listas, uma com os b e outra com os w
    obtidos ao longo da execução (o último valor das listas deve
    corresponder à última época/iteração).

    :param data: np.array - matriz com o conjunto de dados, x na coluna 0 e y na coluna 1
    :param b: float - bias (intercepto da reta)
    :param w: float - peso (inclinacao da reta)
    :param alpha: float - taxa de aprendizado (a.k.a. tamanho do passo)
    :param num_iterations: int - numero de épocas/iterações para executar a descida de gradiente
    :return: list,list - uma lista com os b e outra com os w obtidos ao longo da execução
    """

    b_list = [b]
    w_list = [w]

    for i in range(num_iterations):
        b, w = step_gradient(b, w, data, alpha)
        b_list.append(b)
        w_list.append(w)
        print(f"Iteração {i+1}/{num_iterations}: b = {b}, w = {w}, MSE = {compute_mse(b, w, data)}")

    return b_list, w_list

#import data from csv
def load_data(file_path):
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
    return data

from random import randint
bs, ws = fit(load_data('alegrete.csv'), randint(0, 1), randint(0, 1), 0.01, 1000)
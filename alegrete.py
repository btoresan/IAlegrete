import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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

def plot_regression(data, b, w, title="Linear Regression Fit"):
    """
    Plots 2D data and the fitted regression line
    
    Args:
        data: np.array with x in column 0, y in column 1
        b: float - bias term (intercept)
        w: float - weight (slope)
        title: str - plot title
    """
    plt.figure(figsize=(10, 6))
    
    # Extract x and y from data
    x = data[:, 0]
    y = data[:, 1]
    
    # Plot original data
    plt.scatter(x, y, color='blue', label='Data points')
    
    # Create regression line
    x_min, x_max = np.min(x), np.max(x)
    x_line = np.linspace(x_min, x_max, 100)
    y_line = w * x_line + b
    
    # Plot regression line
    plt.plot(x_line, y_line, 'r-', linewidth=2, label=f'Regression line: y = {w:.2f}x + {b:.2f}')
    
    # Add labels and legend
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def animate_training(data, b_history, w_history):
    """
    Displays an animation of the regression line evolving during training
    (for Jupyter notebooks or interactive Python environments)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract x and y from data
    x = data[:, 0]
    y = data[:, 1]
    
    # Plot original data
    ax.scatter(x, y, color='blue', label='Data points')
    
    # Setup for animation
    x_min, x_max = np.min(x), np.max(x)
    x_line = np.linspace(x_min, x_max, 100)
    line, = ax.plot([], [], 'r-', linewidth=2, label='Regression line')
    
    # Add parameter display text
    param_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                        bbox=dict(facecolor='white', alpha=0.8))
    
    # Add epoch counter
    epoch_text = ax.text(0.8, 0.95, '', transform=ax.transAxes,
                        bbox=dict(facecolor='white', alpha=0.8))
    
    def init():
        line.set_data([], [])
        param_text.set_text('')
        epoch_text.set_text('')
        return line, param_text, epoch_text
    
    def update(frame):
        # Update regression line
        current_w = w_history[frame]
        current_b = b_history[frame]
        y_line = current_w * x_line + current_b
        line.set_data(x_line, y_line)
        
        # Update text
        param_text.set_text(f'y = {current_w:.4f}x + {current_b:.4f}')
        epoch_text.set_text(f'Epoch: {frame}/{len(b_history)-1}')
        
        return line, param_text, epoch_text
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=len(b_history),
                       init_func=init, blit=True, interval=50, repeat=False)
    
    # Set plot properties
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(min(y)-1, max(y)+1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Live Training Visualization')
    ax.legend(loc='upper left')
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return ani  # Return the animation object to prevent garbage collection

#import data from csv
def load_data(file_path):
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
    return data

#bs, ws = fit(load_data('alegrete.csv'), 0, 0, 0.01, 1000)

#animate_training(load_data('alegrete.csv'), bs, ws)
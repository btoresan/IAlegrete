# IAlegrete
Trabalho da cadeira de Inteligencia Artifical (2025/1)

## 📦 Dependencias
```python
import numpy as np 
import matplotlib.pyplot as plt
```

## Parametros Utilizados
```python
b_initial = 0.0  
w_initial = 0.0
alpha = 0.001 
iterations = 1000
```

## Extras
Incluimos duas funcoes adicionais em alegrete .py para a visualizacao do treinamento'
```python
 #Plota a reta e os pontos de dados
plot_regression(data, b, w, title="Linear Regression Fit")
#Plota os dados e anima a mudanca da reta ao longo do treinamento
animate_training(load_data('alegrete.csv'), bs, ws) 
```

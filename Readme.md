# Trabalho 1 - Aprendizado Supervisionado
Inteligencia Artifical (2025/1) - Turma B

### Integrantes
- Bernardo Toresan - 00579107
- João Pastorello - 00580242
- Matheus Manica - 00578366

## Exercício 1 - Regressão Linear

### Parâmetros Utilizados
```python
b = 0.0
w = 0.0
alpha = 0.01
num_iterations = 20
```

### Erro quadrático médio obtido
10.48

### Extras


### Conclusões


## Exercício 2 - Tensorflow/Keras

### Análise dos datasets

#### MNIST
- Classes: 10
- Amostras
- - Treino: 60.000
- - Teste: 10.000
- Tamanho das imagens
- - Altura: 28 pixels
- - Largura: 28 pixels
- - Canais de cor: 1 (escala de cinza)

#### Fashion MNIST
- Classes: 10
- Amostras
- - Treino: 60.000
- - Teste: 10.000
- Tamanho das imagens
- - Altura: 28 pixels
- - Largura: 28 pixels
- - Canais de cor: 1 (escala de cinza)

#### CIFAR-10
- Classes: 10
- Amostras
- - Treino: 50.000
- - Teste: 10.000
- Tamanho das imagens
- - Altura: 32 pixels
- - Largura: 32 pixels
- - Canais de cor: 3 (RGB)

#### CIFAR-100
- Classes: 100
- Amostras
- - Treino: 50.000
- - Teste: 10.000
- Tamanho das imagens
- - Altura: 32 pixels
- - Largura: 32 pixels
- - Canais de cor: 3 (RGB)

### Conclusões

#### MNIST
É o dataset mais simples, possuindo imagens 28x28 monocromáticas de apenas 10 classes. Os dígitos indo-arábicos que definem as classes naturalmente possuem diferenças notáveis, apesar das variações presentes na escrita, o que facilita a identificação.

Obtivemos uma acurácia de 97,43%.

#### Fashion MNIST
É um dataset mais difícil que o MNIST, mas ainda simples. Com a mesma estrutura das imagens (28x28 monocromáticas) e número de classes (10), o Fashion MNIST apresenta um desafio maior que o MNIST devido à temática "fashion", em que a diferença entre classes não é tão clara como no MNIST.

Obtivemos uma acurácia de 89,34%.

#### CIFAR-10
É um dataset mais complexo que o MNIST e Fashion MNIST por alguns motivos. As imagens possuem 3 canais de cor e são um pouco maiores, em 32x32. Além disso, as classes possuem desafios conhecidos até para humanos, como a diferenciação de cachorros e gatos.

Obtivemos uma acurácia de 74,73%. Houve melhora na acurácia utilizando as técnincas de normalização e dropout. A normalização foi ligeiramente mais efetiva antes da ativação e o dropout serviu para evitar overfitting.

#### CIFAR-100
É o dataset mais difícil dentre os avaliados. Com a mesma estrutura das imagens do CIFAR-10 (32x32 tricromáticas), o CIFAR-100 se destaca por suas 100 classes, que podem ser agrupadas em 20 superclasses. Esse grande número de classes dificulta muito sua classificação.

Obtivemos uma acurácia de 53,76%. Conseguimos grande melhora na acurácia utilizando as técnincas de normalização e dropout. A normalização foi ligeiramente mais efetiva antes da ativação e o dropout serviu para evitar overfitting.

### Extras

Para acelerar o treinamento e avaliação das redes neurais, instalamos e configuramos o Tensorflow para a execução em GPU. Foi utilizada a GPU AMD RX 7600 e o container docker rocm/tensorflow na última versão disponível no momento de instalação ([rocm6.3.3-py3.12-tf2.17-dev](https://hub.docker.com/layers/rocm/tensorflow/rocm6.3.3-py3.12-tf2.17-dev/images/sha256-fd2653f436880366cc874aa24264ca9dabd892d76ccb63fb807debba459bcaaf)), seguindo instruções disponíveis no [site da AMD](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/tensorflow-install.html#using-a-docker-image-with-tensorflow-pre-installed).

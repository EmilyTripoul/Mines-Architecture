# Benchmark pour le cours d'architecture logicielle et matérielle des Mines de Paris (Q4 2018)

L'application calcule la fonction suivante :
$$r=\frac{1}{\sum_{i=0}^n v_i}\sum_{i=0}^n (u_i*v_i-a)^k$$

Pour U et V deux tableaux de flotants.

L'application fonctionne pour k<=0

## Installation

Le téléchargement du repo git s'effectue grâce à la commande
```shell
git clone https://github.com/EmilyTripoul/Mines-Architecture
```

L'application fonctionne sur Windows et Linux. 
Elle utilise CMake pour la phase de compilation, et le compilateur par défaut configuré.
L'outil CMake est trouvable sur le lien suivant : https://cmake.org/download/

CMake est initialisé par la commande suivante :
```shell
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=E:/Logiciels/CUDA/bin/nvcc.exe
```

Si CUDA est détecté par CMake, deux benchmark supplémentaires sont fournis.
La dernière version de CUDA est trouvable sur le lien suivant : https://developer.nvidia.com/cuda-downloads

Pour spécifier à CMake le répertoire du projet, la variable `-DCMAKE_CUDA_COMPILER` peut être utilisée:
```shell
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=C:/CUDA10.0/bin/nvcc.exe
```

La compilation est ensuite réalisée avec la commande
```shell
make --target all
```

La compilation produit deux executables : 
- `Mines_Archi` où l'allignement n'est pas forcé
- `Mines_Archi_aligned` où l'allignement est forcé


## Utilisation

Pour lancer l'application, il faut lancer le fichier `Mines_Archi` ou `Mines_Archi_aligned`dans la console en précisant les paramètres `n k a` :
```
> Mines_Archi -h
Usage : ./bench n [k] [a] [-h] [-run <int>]
        n : unsigned int > 0
        k : int (default = 1)
        a : float (default = 0)
        -h : help
        -run <int> : run number (default = 10)
        -nth <int> : thread number (default = 0)
```

Exemple : 
```
> Mines_Archi 10000000 150 0 -run 10
n=10000000      k=150   a=0
Benchmark       CPU time (ms)           Wall time (ms)  Result
                min     max     avg     min     max     avg
=============================================================================
SEQUENTIAL
ipow            203.00  208.00  204.90  203.25  207.30  204.90 (0.00%)  0.01
-----------------------------------------------------------------------------
VECTORIAL
AVX             116.00  117.00  116.70  115.85  117.14  116.71 (75.57%) 0.01
AVX_par         17.00   19.00   18.00   17.65   18.44   18.02 (1037.34%)        0.01
-----------------------------------------------------------------------------
PARALLEL
openMP          32.00   35.00   32.90   31.83   34.57   32.89 (522.99%) 0.01
std_thread      32.00   34.00   33.00   32.49   34.22   33.04 (520.20%) 0.01
std_thread_atom 32.00   37.00   33.10   32.04   36.61   33.10 (519.11%) 0.01
-----------------------------------------------------------------------------
CUDA
cuda            37.00   43.00   40.20   37.15   42.47   40.21 (409.55%) 0.01
cuda_op         17.00   19.00   17.90   16.35   19.13   17.90 (1044.53%)        0.01
=============================================================================
```


## Test
L'application a été testée sur Windows 10 avec le compilateur MSVC 2017 et CUDA 10.0 ainsi que Ubuntu 18.04 avec gcc 6 et CUDA 9.

Deux fichiers `run_test` ont été fournis pour simplifier les tests.

Les tests sont les suivants :
- ipow : Calcul de la puissance par un algorithme de puissance optimisé
- AVX : Améliore le calcul précédent grâce à l'AVX-512
- AVX_par : Version multi-thread de AVX via des atomiques pour la réduction
- OpenMP : Version multi-thread via openMP
- std_thread : Utilise std_thread et des mutex
- std_thread_atom : Utilise std_thread et des atomiques
- cuda : Utilise CUDA la réduction n'est pas optimisée
- cuda_op : Utilise CUDA, la réduction est optimisée pour des blocksize <= 1024

По определению (в компонентном виде) $\delta^l_k = \frac{\partial J}{\partial z_k^l}$  
$J$ можно рассматривать как функцию от $z_{j}^{l+1}$, которые, в свою очередь функции от $z_i^l$  
$ z_k^{l+1}(z^l_1,...) = \sum_i w^{l+1}_{ki} \sigma(z^l_i) + b^{l+1}_k $  
Воспользуемся правилом дифференцирования сложной функции:  
$\delta^l_k = \frac{\partial J}{\partial z_k^l} = \sum_j \frac{\partial J}{\partial z^{l+1}_j} \cdot \frac{\partial z^{l+1}_j}{\partial z^l_k} = \sum_j \delta^{l+1}_j \cdot w^{l+1}_{jk} \cdot \sigma'(z^l_k)$  
Теперь осталось переписать это в векторном виде:  
$\delta^{l} = ((w^{l+1})^T\delta^{l+1}) \odot \sigma'(z^l)$  

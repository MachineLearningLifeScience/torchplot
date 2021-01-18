![Logo](logo.png)

---

# torchplot - Plotting pytorch tensors made easy!

Ask yourself the following:
* Are you using `matplotlib.pyplot` to plot pytorch tensors?
* Do you forget to call `.cpu().detach().numpy()` everytime you want to plot a tensor

Then `torchplot` may be something for you. `torchplot` is a simple drop-in replacement
for plotting pytorch tensors. We simply override every `matplotlib.pyplot` function such
that pytorch tensors are automatically converted.

Simply just change your default `matplotlib` import statement:


Instead of
```
from matplotlib.pyplot import *
```
use
```
from torchplot import *
```
and instead of
```
import matplotlib.pyplot as plt
```
use
```
import torchplot as plt
```
Herafter, then you can remove every `.cpu().detach().numpy()` (or variations heroff) from
your code and everything should just work. If you do not want to mix implementations, 
we recommend importing `torchplot` as seperaly package:
```
import torchplot as tp
```

## Installation
Simple as
```
pip install torchplot
```

## Example

``` python
# lets make a scatter plot of two pytorch variables that are stored on gpu
import torch
import torchplot as plt
x = torch.randn(100, requires_grad=True, device='cuda')
y = torch.randn(100, requires_grad=True, device='cuda')
plt.plot(x, y, '.') # easy and simple
```







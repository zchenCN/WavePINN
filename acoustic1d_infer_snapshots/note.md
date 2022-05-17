
## Settings
- num_init = 1000, num_pde = 1000, wp=0.0001, wi=1.0, wb=0.0, good results?
- num_init = 1000, num_pde = 1000, wp=1.0, wi=1.0, wb=0.0, bad results
- num_init = 1000, num_pde = 1000, wp=0.1, wi=1.0, wb=0.0, bad results
- num_init = 1000, num_pde = 1000, wp=0.01, wi=1.0, wb=0.0, bad results
- num_init = 1000, num_pde = 1000, wp=0.001, wi=1.0, wb=0.0, bad results
- num_init = 1000, num_pde = 1000, wp=0.00001, wi=1.0, wb=0.0, good results ramdom sample
- num_init = 1000, num_pde = 10000, wp=0.00001, wi=1.0, wb=0.0, bad results
- num_init = 1000, num_pde = 2000, wp=0.00001, wi=1.0, wb=0.0, not bad results

| wp|lr | result|
|:---:|:---:|:---:|
|0.1|0.001|0.78|
|0.01|0.001|0.76|
|0.001|0.001|0.83|
|0.0001|0.001|0.83|
|0.00001|0.001|0.2|
|0.000001|0.001|0.04|
|0.0|0.001||0.8|
|0.000003|0.001||

|num_pde|$l_2$ error|
|:---:|:---:|
|1000 random|0.04|
|5000 random|0.01|
|10000 random|0.015|


## Problems
- How dose number of collocations points influence the process of training
- How dose the method of sampling collocations points influence the process of training
# 5604

## Setup:
Install requirements from `requirements.txt`
```
conda create -n <environment-name> --file requirements.txt
```


## Test:
On Vimeo90K:

- Download dataset from http://toflow.csail.mit.edu/
- Extract data in `data/vimeo90k_triplet`
- Run: 
```
python test.py --model_name [path/to/pretrained/weights]
```    






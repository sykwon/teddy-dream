# Traning Data Generation

## Prerequisite

```bash
sudo apt-get install redis-server
sudo apt-get install binutils
pip install redis
```

## Script

```bash
# quick test
./run.sh test 

# exp on DBLP 
./run.sh dblp

# all exp
./run.sh all
```

## Descriptions

### Algorithms

| in exp | in paper |
|--------|----------|
| allp   | NavieGen |
| topk   | Qgram    |
| taste  | TASTE    |
| soddy2 | SODDY    |
| teddy2 | TEDDY    |
| teddy0 | TEDDY-S  |
| abl1   | TEDDY-R  |

### Datasets

| in exp | in paper |
|--------|----------|
| dblp   |   DBLP   |
| wiki2  |   WIKI   |
| imdb2  |   IMDB   |
| egr1   |   GENE   |

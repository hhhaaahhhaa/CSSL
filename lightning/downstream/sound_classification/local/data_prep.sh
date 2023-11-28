src=$1
dst=$2
ds=$3

. ./path.sh
python local/preprocess.py $src $dst --dataset $ds --create_dataset

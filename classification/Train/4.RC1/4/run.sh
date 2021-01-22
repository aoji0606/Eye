sh train.sh

cp ./model/model_19.h5 ./model/shared_model.h5

sh shared_train.sh

sh test.sh

python vote.py >> result.txt
python probability.py >> result.txt

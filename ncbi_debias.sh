value1="BiasProduct"
value2="Learned-Mixin"
value3="Reweight"
DATA="NCBI_disease"


echo "Running $value1"
echo "Model - early stopped"
echo "Dataset - $DATA"
echo ""

python biobert_clf_debias.py \
    --dataset_name $DATA \
    --loss_fn $value1 \
    --bias_file ${DATA}_prior_prob.txt \
    --model_name model 

echo "Generating prediction file using debiased-early stopped model"

python inference.py \
    --dataset_name $DATA \
    --model_name model_${value1} \
    --loss_fn $value1

echo ""
echo "Performance on $DATA using debiased-early stopped model"
echo ""

python evaluate.py \
    --mention_dictionary datasets/$DATA/mention_dict.txt \
    --cui_dictionary datasets/$DATA/cui_dict.txt \
    --gold_labels datasets/$DATA/test.txt \
    --gold_cuis datasets/$DATA/test_cuis.txt \
    --predictions resources/$DATA/preds_model_${value1}.txt 


echo "Running $value2"
echo "Model - early stopped"
echo "Dataset - $DATA"
echo ""

python biobert_clf_debias.py \
    --dataset_name $DATA \
    --loss_fn $value2 \
    --bias_file ${DATA}_prior_prob.txt \
    --model_name model 

echo "Generating prediction file using debiased-early stopped model"

python inference.py \
    --dataset_name $DATA \
    --model_name model_${value2} \
    --loss_fn $value2

echo ""
echo "Performance on $DATA using debiased-early stopped model"
echo ""

python evaluate.py \
    --mention_dictionary datasets/$DATA/mention_dict.txt \
    --cui_dictionary datasets/$DATA/cui_dict.txt \
    --gold_labels datasets/$DATA/test.txt \
    --gold_cuis datasets/$DATA/test_cuis.txt \
    --predictions resources/$DATA/preds_model_${value2}.txt 


echo "Running $value3"
echo "Model - early stopped"
echo "Dataset - $DATA"
echo ""

python biobert_clf_debias.py \
    --dataset_name $DATA \
    --loss_fn $value3 \
    --bias_file ${DATA}_prior_prob.txt \
    --model_name model 

echo "Generating prediction file using debiased-early stopped model"

python inference.py \
    --dataset_name $DATA \
    --model_name model_${value3} \
    --loss_fn $value3

echo ""
echo "Performance on $DATA using debiased-early stopped model"
echo ""

python evaluate.py \
    --mention_dictionary datasets/$DATA/mention_dict.txt \
    --cui_dictionary datasets/$DATA/cui_dict.txt \
    --gold_labels datasets/$DATA/test.txt \
    --gold_cuis datasets/$DATA/test_cuis.txt \
    --predictions resources/$DATA/preds_model_${value3}.txt 

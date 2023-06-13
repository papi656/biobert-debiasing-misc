DATA="NCBI_disease"

echo "Running Confidence regularization"
echo "Model - early stopped"
echo "Dataset - $DATA"
echo ""

python biobert_clf_debias.py \
    --dataset_name $DATA \
    --loss_fn Confidence_regularization \
    --bias_file ${DATA}_prior_prob.txt \
    --model_name model 

echo "Generating prediction file using debiased-early stopped model"

python inference.py \
    --dataset_name $DATA \
    --model_name model_Confidence_regularization \
    --loss_fn Confidence_regularization

echo ""
echo "Performance on $DATA using debiased-early stopped model"
echo ""

python evaluate.py \
    --mention_dictionary datasets/$DATA/mention_dict.txt \
    --cui_dictionary datasets/$DATA/cui_dict.txt \
    --gold_labels datasets/$DATA/test.txt \
    --gold_cuis datasets/$DATA/test_cuis.txt \
    --predictions resources/$DATA/preds_model_Confidence_regularization.txt 

